import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# =========================================================
# CONFIG
# =========================================================
VIDEO_SPLITS_PATH = Path("data/slowfast_video_only/video_only_splits.csv")
KINE_X_PATH = Path("X_ts.npy")
KINE_META_PATH = Path("meta.csv")
KINE_CTX_PATH = Path("X_ctx.csv")
KINE_MODEL_PATH = Path("best_model.pth")

CACHE_DIR = Path("data/slowfast_midlevel_fusion_hierarchical")
KINE_EMBED_DIR = CACHE_DIR / "kine_embeddings"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
KINE_EMBED_DIR.mkdir(parents=True, exist_ok=True)

FUSION_META_PATH = CACHE_DIR / "midlevel_fusion_hierarchical_splits.csv"
STAGE1_MODEL_PATH = CACHE_DIR / "stage1_sce_binary_best.pth"
STAGE2_MODEL_PATH = CACHE_DIR / "stage2_sce_type_best.pth"
PRED_PATH = CACHE_DIR / "hierarchical_test_predictions.csv"

BATCH_SIZE = 64
EPOCHS = 40
PATIENCE = 10
LEARNING_RATE = 1e-3
SEED = 1
NUM_CLASSES_FINAL = 4
NUM_CLASSES_STAGE1 = 2
NUM_CLASSES_STAGE2 = 3
NOT_SCE_IDX = 3

CLASS_MAP_STR = {
    0: "Conflict",
    1: "Bump",
    2: "Hard Brake",
    3: "Not an SCE",
}

STAGE1_MAP_STR = {
    0: "Not an SCE",
    1: "SCE",
}

STAGE2_MAP_STR = {
    0: "Conflict",
    1: "Bump",
    2: "Hard Brake",
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {DEVICE.type.upper()} device!")


# =========================================================
# REPRODUCIBILITY
# =========================================================
def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =========================================================
# KINEMATICS / CONTEXT MODEL FROM NOTEBOOK
# =========================================================
class MultiModal_CNN_LSTM(nn.Module):
    def __init__(self, in_chans_ts: int, in_chans_ctx: int, num_classes: int) -> None:
        super().__init__()
        self.bn_input = nn.BatchNorm1d(in_chans_ts)
        self.conv1 = nn.Conv1d(in_channels=in_chans_ts, out_channels=64, kernel_size=12, stride=1, padding="same")
        self.bn_conv = nn.BatchNorm1d(64)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.lstm = nn.LSTM(input_size=64, hidden_size=100, num_layers=2, batch_first=True, dropout=0.3)

        self.ctx_fc1 = nn.Linear(in_chans_ctx, 64)
        self.ctx_relu = nn.ReLU()
        self.ctx_bn = nn.BatchNorm1d(64)
        self.ctx_drop = nn.Dropout(0.3)

        fusion_size = 100 + 64
        self.fusion_drop = nn.Dropout(0.5)
        self.fc_final = nn.Linear(fusion_size, num_classes)

    def extract_fused_features(self, x_ts: torch.Tensor, x_ctx: torch.Tensor) -> torch.Tensor:
        x_ts = self.bn_input(x_ts)
        x_ts = self.conv1(x_ts)
        x_ts = self.bn_conv(x_ts)
        x_ts = self.relu1(x_ts)
        x_ts = self.maxpool1(x_ts)
        lstm_out, _ = self.lstm(x_ts.permute(0, 2, 1))
        x_ts_features, _ = torch.max(lstm_out, dim=1)

        x_ctx_features = self.ctx_fc1(x_ctx)
        x_ctx_features = self.ctx_bn(x_ctx_features)
        x_ctx_features = self.ctx_relu(x_ctx_features)
        x_ctx_features = self.ctx_drop(x_ctx_features)

        fused = torch.cat((x_ts_features, x_ctx_features), dim=1)
        return fused

    def forward(self, x_ts: torch.Tensor, x_ctx: torch.Tensor) -> torch.Tensor:
        fused = self.extract_fused_features(x_ts, x_ctx)
        out = self.fusion_drop(fused)
        out = self.fc_final(out)
        return out


# =========================================================
# HIERARCHICAL FUSION MODEL
# =========================================================
class FusionBackbone(nn.Module):
    def __init__(self, video_dim: int = 2304, kine_dim: int = 164) -> None:
        super().__init__()
        self.video_proj = nn.Sequential(
            nn.Linear(video_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
        )
        self.kine_proj = nn.Sequential(
            nn.Linear(kine_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 256),
            nn.ReLU(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(256 * 4, 512),
            nn.ReLU(),
            nn.Dropout(0.35),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(256, 128),
            nn.ReLU(),
        )

    def forward(self, video_embedding: torch.Tensor, kine_embedding: torch.Tensor) -> torch.Tensor:
        v = self.video_proj(video_embedding)
        k = self.kine_proj(kine_embedding)
        joint = torch.cat([v, k, v * k, torch.abs(v - k)], dim=1)
        return self.fusion(joint)


class HierarchicalFusionClassifier(nn.Module):
    def __init__(self, num_classes: int, video_dim: int = 2304, kine_dim: int = 164) -> None:
        super().__init__()
        self.backbone = FusionBackbone(video_dim=video_dim, kine_dim=kine_dim)
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, video_embedding: torch.Tensor, kine_embedding: torch.Tensor) -> torch.Tensor:
        fused = self.backbone(video_embedding, kine_embedding)
        return self.classifier(fused)


# =========================================================
# DATASETS
# =========================================================
class KineFeatureDataset(Dataset):
    def __init__(self, X_ts: np.ndarray, X_ctx: np.ndarray, bdd_ids: np.ndarray) -> None:
        self.X_ts = torch.tensor(X_ts, dtype=torch.float32)
        self.X_ctx = torch.tensor(X_ctx, dtype=torch.float32)
        self.bdd_ids = list(map(str, bdd_ids))

    def __len__(self) -> int:
        return len(self.bdd_ids)

    def __getitem__(self, idx: int):
        return self.X_ts[idx], self.X_ctx[idx], self.bdd_ids[idx]


class MidLevelFusionDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, target_col: str) -> None:
        self.df = dataframe.reset_index(drop=True)
        self.target_col = target_col

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        video_emb = torch.load(row["embedding_path"], map_location="cpu", weights_only=False).float()
        kine_emb = torch.load(row["kine_embedding_path"], map_location="cpu", weights_only=False).float()
        target = int(row[self.target_col])
        bdd_id = str(row["BDD_ID"])
        return video_emb, kine_emb, target, bdd_id


# =========================================================
# PREP / MERGE
# =========================================================
def add_hierarchical_targets(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["stage1_target"] = (df["target_idx"] != NOT_SCE_IDX).astype(int)
    df["stage2_target"] = df["target_idx"].astype(int)
    return df



def load_and_align_data() -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    video_df = pd.read_csv(VIDEO_SPLITS_PATH)
    meta = pd.read_csv(KINE_META_PATH)
    ctx = pd.read_csv(KINE_CTX_PATH)
    X_ts = np.load(KINE_X_PATH).astype(np.float32)

    assert len(meta) == len(ctx) == len(X_ts), "Mismatch among meta, ctx, and X_ts lengths"
    assert "BDD_ID" in meta.columns, "meta.csv must contain BDD_ID"
    assert "y" in meta.columns, "meta.csv must contain zero-indexed y labels"

    meta = meta.copy()
    meta["BDD_ID"] = meta["BDD_ID"].astype(str)
    meta["row_idx"] = np.arange(len(meta))

    video_df = video_df.copy()
    video_df["BDD_ID"] = video_df["BDD_ID"].astype(str)

    merged = video_df.merge(meta[["BDD_ID", "row_idx", "y"]], on="BDD_ID", how="inner")
    merged = merged.drop_duplicates(subset=["BDD_ID"]).reset_index(drop=True)

    if "target_idx" not in merged.columns:
        merged["target_idx"] = merged["y"].astype(int)

    ctx_features = ctx.drop(columns=[c for c in ["BDD_ID", "EVENT_ID", "EVENT_TYPE", "y"] if c in ctx.columns])
    categorical_cols = [c for c in ["weather", "scene", "timeofday"] if c in ctx_features.columns]
    ctx_features = pd.get_dummies(ctx_features, columns=categorical_cols, dummy_na=True)
    ctx_features = ctx_features.fillna(0)
    X_ctx_all = ctx_features.to_numpy(dtype=np.float32)

    row_idx = merged["row_idx"].to_numpy(dtype=int)
    X_ts_sel = X_ts[row_idx]
    X_ctx_sel = X_ctx_all[row_idx]

    train_mask = merged["split"].to_numpy() == "train"
    scaler = StandardScaler()
    X_ctx_sel[train_mask] = scaler.fit_transform(X_ctx_sel[train_mask])
    X_ctx_sel[~train_mask] = scaler.transform(X_ctx_sel[~train_mask])

    merged = add_hierarchical_targets(merged)
    return merged, X_ts_sel, X_ctx_sel


@torch.no_grad()
def export_kine_embeddings(df: pd.DataFrame, X_ts: np.ndarray, X_ctx: np.ndarray) -> pd.DataFrame:
    if not KINE_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Could not find {KINE_MODEL_PATH}. Rerun the notebook first so best_model.pth exists."
        )

    print("Exporting kinematics/context embeddings...")
    ds = KineFeatureDataset(X_ts, X_ctx, df["BDD_ID"].to_numpy())
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=False)

    model = MultiModal_CNN_LSTM(
        in_chans_ts=X_ts.shape[1],
        in_chans_ctx=X_ctx.shape[1],
        num_classes=NUM_CLASSES_FINAL,
    ).to(DEVICE)
    model.load_state_dict(torch.load(KINE_MODEL_PATH, map_location=DEVICE, weights_only=False))
    model.eval()

    embed_map: Dict[str, str] = {}
    for batch_ts, batch_ctx, batch_ids in tqdm(loader, desc="Exporting kinematics/context embeddings"):
        batch_ts = batch_ts.to(DEVICE)
        batch_ctx = batch_ctx.to(DEVICE)
        fused = model.extract_fused_features(batch_ts, batch_ctx).detach().cpu()
        for i, bdd_id in enumerate(batch_ids):
            out_path = KINE_EMBED_DIR / f"{bdd_id}.pt"
            if not out_path.exists():
                torch.save(fused[i], out_path)
            embed_map[str(bdd_id)] = str(out_path)

    out_df = df.copy()
    out_df["kine_embedding_path"] = out_df["BDD_ID"].map(embed_map)
    out_df = out_df.dropna(subset=["embedding_path", "kine_embedding_path"]).reset_index(drop=True)
    out_df.to_csv(FUSION_META_PATH, index=False)
    print(f"Saved hierarchical fusion metadata to: {FUSION_META_PATH}")
    return out_df


# =========================================================
# TRAIN / EVAL HELPERS
# =========================================================
def make_loaders(df: pd.DataFrame, target_col: str, stage2_only: bool = False) -> Tuple[DataLoader, DataLoader, DataLoader]:
    work_df = df.copy()
    if stage2_only:
        work_df = work_df[work_df["stage1_target"] == 1].copy()

    train_df = work_df[work_df["split"] == "train"].copy()
    val_df = work_df[work_df["split"] == "val"].copy()
    test_df = work_df[work_df["split"] == "test"].copy()

    train_loader = DataLoader(MidLevelFusionDataset(train_df, target_col=target_col), batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=False)
    val_loader = DataLoader(MidLevelFusionDataset(val_df, target_col=target_col), batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=False)
    test_loader = DataLoader(MidLevelFusionDataset(test_df, target_col=target_col), batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=False)
    return train_loader, val_loader, test_loader



def evaluate_loader(model: nn.Module, loader: DataLoader) -> Tuple[float, List[int], List[int], List[str]]:
    model.eval()
    y_true: List[int] = []
    y_pred: List[int] = []
    ids: List[str] = []

    with torch.no_grad():
        for batch_v, batch_k, batch_targets, batch_ids in loader:
            batch_v = batch_v.to(DEVICE)
            batch_k = batch_k.to(DEVICE)
            logits = model(batch_v, batch_k)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            y_true.extend(batch_targets.numpy().tolist())
            y_pred.extend(preds.tolist())
            ids.extend(list(batch_ids))

    acc = accuracy_score(y_true, y_pred) if len(y_true) > 0 else 0.0
    return acc, y_true, y_pred, ids



def train_single_model(
    df: pd.DataFrame,
    target_col: str,
    num_classes: int,
    model_path: Path,
    stage_name: str,
    stage2_only: bool = False,
) -> HierarchicalFusionClassifier:
    train_loader, val_loader, _ = make_loaders(df, target_col=target_col, stage2_only=stage2_only)
    model = HierarchicalFusionClassifier(num_classes=num_classes).to(DEVICE)

    train_df = df.copy()
    if stage2_only:
        train_df = train_df[train_df["stage1_target"] == 1].copy()
    train_targets = train_df[train_df["split"] == "train"][target_col].to_numpy(dtype=int)

    class_counts = np.bincount(train_targets, minlength=num_classes)
    class_weights = len(train_targets) / (num_classes * np.maximum(class_counts, 1))
    class_weights = torch.tensor(class_weights, dtype=torch.float32, device=DEVICE)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_acc = -1.0
    patience_ctr = 0
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_v, batch_k, batch_targets, _ in tqdm(train_loader, desc=f"{stage_name} Epoch {epoch + 1:02d}/{EPOCHS}", leave=False):
            batch_v = batch_v.to(DEVICE)
            batch_k = batch_k.to(DEVICE)
            batch_targets = batch_targets.to(DEVICE)

            optimizer.zero_grad()
            logits = model(batch_v, batch_k)
            loss = criterion(logits, batch_targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * batch_targets.size(0)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == batch_targets).sum().item()
            total += batch_targets.size(0)

        train_loss = running_loss / max(total, 1)
        train_acc = correct / max(total, 1)
        val_acc, _, _, _ = evaluate_loader(model, val_loader)
        print(f"{stage_name} Epoch {epoch + 1:02d} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_ctr = 0
            torch.save(model.state_dict(), model_path)
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                print(f"{stage_name}: early stopping at epoch {epoch + 1}.")
                break

    print(f"Best {stage_name} model saved to: {model_path}")
    return model



def evaluate_individual_stage(model: nn.Module, df: pd.DataFrame, target_col: str, stage_name: str, label_map: Dict[int, str], stage2_only: bool = False) -> None:
    _, _, test_loader = make_loaders(df, target_col=target_col, stage2_only=stage2_only)
    acc, y_true, y_pred, _ = evaluate_loader(model, test_loader)
    print("\n" + "-" * 80)
    print(f"{stage_name} TEST RESULTS")
    print(f"Test Accuracy: {acc:.4f}")
    print(classification_report(y_true, y_pred, target_names=[label_map[i] for i in range(len(label_map))]))
    print("-" * 80)



def hierarchical_predict(
    stage1_model: nn.Module,
    stage2_model: nn.Module,
    df: pd.DataFrame,
) -> pd.DataFrame:
    test_df = df[df["split"] == "test"].copy().reset_index(drop=True)
    test_loader = DataLoader(MidLevelFusionDataset(test_df, target_col="target_idx"), batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=False)

    rows: List[Dict[str, object]] = []
    stage1_model.eval()
    stage2_model.eval()

    with torch.no_grad():
        for batch_v, batch_k, batch_targets, batch_ids in test_loader:
            batch_v = batch_v.to(DEVICE)
            batch_k = batch_k.to(DEVICE)

            stage1_logits = stage1_model(batch_v, batch_k)
            stage1_preds = torch.argmax(stage1_logits, dim=1).cpu().numpy()

            stage2_logits = stage2_model(batch_v, batch_k)
            stage2_preds = torch.argmax(stage2_logits, dim=1).cpu().numpy()

            for i, bdd_id in enumerate(batch_ids):
                stage1_pred = int(stage1_preds[i])
                stage2_pred = int(stage2_preds[i])
                final_pred = NOT_SCE_IDX if stage1_pred == 0 else stage2_pred
                true_idx = int(batch_targets[i].item())
                rows.append(
                    {
                        "BDD_ID": str(bdd_id),
                        "true_idx": true_idx,
                        "true_label": CLASS_MAP_STR[true_idx],
                        "stage1_true": int(true_idx != NOT_SCE_IDX),
                        "stage1_true_label": STAGE1_MAP_STR[int(true_idx != NOT_SCE_IDX)],
                        "stage1_pred": stage1_pred,
                        "stage1_pred_label": STAGE1_MAP_STR[stage1_pred],
                        "stage2_pred": stage2_pred,
                        "stage2_pred_label": STAGE2_MAP_STR[stage2_pred],
                        "pred_idx": final_pred,
                        "pred_label": CLASS_MAP_STR[final_pred],
                    }
                )

    return pd.DataFrame(rows)



def evaluate_hierarchical(stage1_model: nn.Module, stage2_model: nn.Module, df: pd.DataFrame) -> None:
    pred_df = hierarchical_predict(stage1_model, stage2_model, df)
    y_true = pred_df["true_idx"].to_numpy(dtype=int)
    y_pred = pred_df["pred_idx"].to_numpy(dtype=int)
    acc = accuracy_score(y_true, y_pred)

    print("\n" + "=" * 80)
    print("HIERARCHICAL TEST RESULTS")
    print(f"Final 4-Class Test Accuracy: {acc:.4f}")
    print(classification_report(y_true, y_pred, target_names=[CLASS_MAP_STR[i] for i in range(NUM_CLASSES_FINAL)]))
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES_FINAL))))
    pred_df.to_csv(PRED_PATH, index=False)
    print(f"Saved hierarchical test predictions to: {PRED_PATH}")
    print("=" * 80)


# =========================================================
# MAIN
# =========================================================
def main() -> None:
    set_seed(SEED)
    df, X_ts, X_ctx = load_and_align_data()
    df = export_kine_embeddings(df, X_ts, X_ctx)

    stage1_model = train_single_model(
        df=df,
        target_col="stage1_target",
        num_classes=NUM_CLASSES_STAGE1,
        model_path=STAGE1_MODEL_PATH,
        stage_name="Stage 1 (SCE vs Not-SCE)",
        stage2_only=False,
    )
    stage1_model.load_state_dict(torch.load(STAGE1_MODEL_PATH, map_location=DEVICE, weights_only=False))
    evaluate_individual_stage(stage1_model, df, target_col="stage1_target", stage_name="Stage 1 (Binary)", label_map=STAGE1_MAP_STR, stage2_only=False)

    stage2_model = train_single_model(
        df=df,
        target_col="stage2_target",
        num_classes=NUM_CLASSES_STAGE2,
        model_path=STAGE2_MODEL_PATH,
        stage_name="Stage 2 (SCE Type)",
        stage2_only=True,
    )
    stage2_model.load_state_dict(torch.load(STAGE2_MODEL_PATH, map_location=DEVICE, weights_only=False))
    evaluate_individual_stage(stage2_model, df, target_col="stage2_target", stage_name="Stage 2 (Conflict/Bump/Hard Brake)", label_map=STAGE2_MAP_STR, stage2_only=True)

    evaluate_hierarchical(stage1_model, stage2_model, df)


if __name__ == "__main__":
    main()
