
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# =========================================================
# CONFIG
# =========================================================
SLOWFAST_SPLITS_PATH = Path("data/slowfast_video_only/video_only_splits.csv")
SLOWFAST_MODEL_PATH = Path("data/slowfast_video_only/slowfast_video_head_best.pth")

# Files from the notebook workflow
X_TS_PATH = Path("data/X_ts.npy")
META_PATH = Path("data/meta.csv")
X_CTX_PATH = Path("data/X_ctx.csv")
KINE_MODEL_PATH = Path("best_model.pth")

OUT_DIR = Path("data/slowfast_kine_concat")
KINE_EMBED_DIR = OUT_DIR / "kine_embeddings"
OUT_DIR.mkdir(parents=True, exist_ok=True)
KINE_EMBED_DIR.mkdir(parents=True, exist_ok=True)

MERGED_SPLITS_PATH = OUT_DIR / "fusion_splits.csv"
FUSION_MODEL_PATH = OUT_DIR / "concat_fusion_best.pth"
TEST_PRED_PATH = OUT_DIR / "concat_fusion_test_predictions.csv"

SEED = 1
NUM_CLASSES = 4
BATCH_SIZE = 64
EPOCHS = 30
LEARNING_RATE = 1e-3

CLASS_MAP_STR = {
    0: "Conflict",
    1: "Bump",
    2: "Hard Brake",
    3: "Not an SCE",
}

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print("Using CUDA GPU!")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    print("Using Apple Silicon (MPS) GPU!")
else:
    DEVICE = torch.device("cpu")
    print("Using CPU.")


# =========================================================
# REPRODUCIBILITY
# =========================================================
def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =========================================================
# MODELS
# =========================================================
class MultiModal_CNN_LSTM(nn.Module):
    """
    Recreated from Copy_of_kine_bm_cnn_lstm_3_way.ipynb.
    We use it here to extract the 164-d fused kinematics/context embedding:
        100-d time-series embedding + 64-d context embedding.
    """
    def __init__(self, in_chans_ts: int, seq_len: int, in_chans_ctx: int, num_classes: int):
        super().__init__()
        self.bn_input = nn.BatchNorm1d(in_chans_ts)
        self.conv1 = nn.Conv1d(in_channels=in_chans_ts, out_channels=64, kernel_size=12, stride=1, padding='same')
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

    def extract_features(self, x_ts: torch.Tensor, x_ctx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x_ts = self.bn_input(x_ts)
        x_ts = self.conv1(x_ts)
        x_ts = self.bn_conv(x_ts)
        x_ts = self.relu1(x_ts)
        x_ts = self.maxpool1(x_ts)
        lstm_out, _ = self.lstm(x_ts.permute(0, 2, 1))
        x_ts_features, _ = torch.max(lstm_out, dim=1)  # (B, 100)

        x_ctx_features = self.ctx_fc1(x_ctx)
        x_ctx_features = self.ctx_bn(x_ctx_features)
        x_ctx_features = self.ctx_relu(x_ctx_features)
        x_ctx_features = self.ctx_drop(x_ctx_features)  # (B, 64)

        fused = torch.cat((x_ts_features, x_ctx_features), dim=1)  # (B, 164)
        return x_ts_features, x_ctx_features, fused

    def forward(self, x_ts: torch.Tensor, x_ctx: torch.Tensor) -> torch.Tensor:
        _, _, fused = self.extract_features(x_ts, x_ctx)
        out = self.fusion_drop(fused)
        out = self.fc_final(out)
        return out


class ConcatFusionHead(nn.Module):
    """
    Concatenate:
      - SlowFast video embedding (2304)
      - CNN/LSTM fused kinematics-context embedding (164)
    Then classify with a small MLP.
    """
    def __init__(self, video_dim: int = 2304, kine_dim: int = 164, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(video_dim + kine_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, video_embedding: torch.Tensor, kine_embedding: torch.Tensor) -> torch.Tensor:
        x = torch.cat([video_embedding, kine_embedding], dim=1)
        return self.net(x)


# =========================================================
# DATASETS
# =========================================================
class ConcatFusionDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame):
        self.df = dataframe.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        video_embedding = torch.load(row["embedding_path"], map_location="cpu", weights_only=False).float()
        kine_embedding = torch.load(row["kine_embedding_path"], map_location="cpu", weights_only=False).float()
        target = int(row["target_idx"])
        bdd_id = row["BDD_ID"]
        return video_embedding, kine_embedding, target, bdd_id


# =========================================================
# LOAD / REBUILD KINEMATICS NOTEBOOK DATA
# =========================================================
def load_kine_notebook_data():
    X = np.load(X_TS_PATH).astype(np.float32)
    meta = pd.read_csv(META_PATH).copy()
    ctx = pd.read_csv(X_CTX_PATH).copy()

    if "y" not in meta.columns:
        raise ValueError("meta.csv must contain a 'y' column from the notebook preprocessing.")
    if "BDD_ID" not in meta.columns:
        raise ValueError("meta.csv must contain 'BDD_ID' to align with SlowFast outputs.")

    y = meta["y"].to_numpy(dtype=np.int64)

    groups = meta["BDD_ID"].astype(str).to_numpy()
    gss1 = GroupShuffleSplit(n_splits=1, train_size=0.70, random_state=SEED)
    train_idx, temp_idx = next(gss1.split(X, y, groups=groups))

    X_train, y_train = X[train_idx], y[train_idx]
    X_temp, y_temp = X[temp_idx], y[temp_idx]
    groups_temp = groups[temp_idx]

    gss2 = GroupShuffleSplit(n_splits=1, train_size=0.50, random_state=SEED)
    val_idx_rel, test_idx_rel = next(gss2.split(X_temp, y_temp, groups=groups_temp))

    val_idx = temp_idx[val_idx_rel]
    test_idx = temp_idx[test_idx_rel]

    split = np.array(["train"] * len(meta), dtype=object)
    split[temp_idx] = "temp"
    split[val_idx] = "val"
    split[test_idx] = "test"
    meta["split"] = split

    ctx_features = ctx.drop(columns=["BDD_ID", "EVENT_ID", "EVENT_TYPE", "y"], errors="ignore")
    categorical_cols = [c for c in ["weather", "scene", "timeofday"] if c in ctx_features.columns]
    ctx_features = pd.get_dummies(ctx_features, columns=categorical_cols, dummy_na=True)
    ctx_features = ctx_features.fillna(0)

    X_ctx_np = ctx_features.to_numpy(dtype=np.float32)

    X_ctx_train, X_ctx_temp = X_ctx_np[train_idx], X_ctx_np[temp_idx]
    X_ctx_val, X_ctx_test = X_ctx_temp[val_idx_rel], X_ctx_temp[test_idx_rel]

    scaler = StandardScaler()
    X_ctx_train = scaler.fit_transform(X_ctx_train)
    X_ctx_val = scaler.transform(X_ctx_val)
    X_ctx_test = scaler.transform(X_ctx_test)

    # stitch back together in original order
    X_ctx_scaled = np.zeros_like(X_ctx_np, dtype=np.float32)
    X_ctx_scaled[train_idx] = X_ctx_train
    X_ctx_scaled[val_idx] = X_ctx_val
    X_ctx_scaled[test_idx] = X_ctx_test

    return X, X_ctx_scaled.astype(np.float32), meta


@torch.no_grad()
def export_kine_embeddings_from_saved_model() -> pd.DataFrame:
    if not KINE_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Could not find {KINE_MODEL_PATH}. "
            "The notebook saves only best_model.pth by default, not per-sample embeddings. "
            "Rerun the notebook first or place best_model.pth here."
        )

    X_ts, X_ctx, meta = load_kine_notebook_data()

    in_chans_ts = X_ts.shape[1]
    seq_len = X_ts.shape[2]
    in_chans_ctx = X_ctx.shape[1]
    num_classes = len(np.unique(meta["y"].to_numpy(dtype=np.int64)))

    model = MultiModal_CNN_LSTM(
        in_chans_ts=in_chans_ts,
        seq_len=seq_len,
        in_chans_ctx=in_chans_ctx,
        num_classes=num_classes,
    ).to(DEVICE)
    model.load_state_dict(torch.load(KINE_MODEL_PATH, map_location=DEVICE, weights_only=False))
    model.eval()

    rows = []
    for idx in tqdm(range(len(meta)), desc="Exporting kinematics/context embeddings"):
        bdd_id = str(meta.iloc[idx]["BDD_ID"])
        x_ts = torch.tensor(X_ts[idx:idx+1], dtype=torch.float32, device=DEVICE)
        x_ctx = torch.tensor(X_ctx[idx:idx+1], dtype=torch.float32, device=DEVICE)

        _, _, fused = model.extract_features(x_ts, x_ctx)
        fused = fused.squeeze(0).detach().cpu()
        out_path = KINE_EMBED_DIR / f"{bdd_id}.pt"
        torch.save(fused, out_path)

        rows.append({
            "BDD_ID": bdd_id,
            "split": meta.iloc[idx]["split"],
            "target_idx": int(meta.iloc[idx]["y"]),
            "kine_embedding_path": str(out_path),
        })

        del x_ts, x_ctx, fused
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()
        elif DEVICE.type == "mps":
            torch.mps.empty_cache()

    return pd.DataFrame(rows)


# =========================================================
# MERGE SLOWFAST + KINEMATICS EMBEDDINGS
# =========================================================
def build_fusion_dataframe() -> pd.DataFrame:
    if not SLOWFAST_SPLITS_PATH.exists():
        raise FileNotFoundError(
            f"Missing {SLOWFAST_SPLITS_PATH}. Run the video-only SlowFast pipeline first."
        )

    slowfast_df = pd.read_csv(SLOWFAST_SPLITS_PATH).copy()
    slowfast_df["BDD_ID"] = slowfast_df["BDD_ID"].astype(str)

    kine_df = export_kine_embeddings_from_saved_model()
    kine_df["BDD_ID"] = kine_df["BDD_ID"].astype(str)

    # Merge on BDD_ID. Prefer the SlowFast split/labels so the concat run follows the video baseline split.
    merged = slowfast_df.merge(
        kine_df[["BDD_ID", "kine_embedding_path"]],
        on="BDD_ID",
        how="inner"
    )

    if len(merged) == 0:
        raise ValueError("No overlapping BDD_ID values found between SlowFast and kinematics outputs.")

    merged.to_csv(MERGED_SPLITS_PATH, index=False)
    print(f"Saved fusion metadata to: {MERGED_SPLITS_PATH}")
    return merged


# =========================================================
# TRAIN / EVAL
# =========================================================
def make_loaders(df: pd.DataFrame):
    train_df = df[df["split"] == "train"].copy()
    val_df = df[df["split"] == "val"].copy()
    test_df = df[df["split"] == "test"].copy()

    train_loader = DataLoader(ConcatFusionDataset(train_df), batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=False)
    val_loader = DataLoader(ConcatFusionDataset(val_df), batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=False)
    test_loader = DataLoader(ConcatFusionDataset(test_df), batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=False)
    return train_loader, val_loader, test_loader


def evaluate_loader(model: nn.Module, loader: DataLoader):
    model.eval()
    y_true, y_pred, ids = [], [], []

    with torch.no_grad():
        for batch_vid, batch_kine, batch_targets, batch_ids in loader:
            batch_vid = batch_vid.to(DEVICE)
            batch_kine = batch_kine.to(DEVICE)
            logits = model(batch_vid, batch_kine)
            preds = torch.argmax(logits, dim=1).cpu().numpy()

            y_true.extend(batch_targets.numpy().tolist())
            y_pred.extend(preds.tolist())
            ids.extend(list(batch_ids))

    acc = accuracy_score(y_true, y_pred)
    return acc, y_true, y_pred, ids


def train_concat_fusion(df: pd.DataFrame) -> ConcatFusionHead:
    train_loader, val_loader, _ = make_loaders(df)
    model = ConcatFusionHead().to(DEVICE)

    train_targets = df[df["split"] == "train"]["target_idx"].to_numpy()
    class_counts = np.bincount(train_targets, minlength=NUM_CLASSES)
    class_weights = len(train_targets) / (NUM_CLASSES * np.maximum(class_counts, 1))
    class_weights = torch.tensor(class_weights, dtype=torch.float32, device=DEVICE)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_acc = -1.0
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        total = 0
        correct = 0

        for batch_vid, batch_kine, batch_targets, _ in tqdm(train_loader, desc=f"Fusion Epoch {epoch + 1}/{EPOCHS}", leave=False):
            batch_vid = batch_vid.to(DEVICE)
            batch_kine = batch_kine.to(DEVICE)
            batch_targets = batch_targets.to(DEVICE)

            optimizer.zero_grad()
            logits = model(batch_vid, batch_kine)
            loss = criterion(logits, batch_targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * batch_targets.size(0)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == batch_targets).sum().item()
            total += batch_targets.size(0)

            del logits, loss, preds
            if DEVICE.type == "cuda":
                torch.cuda.empty_cache()
            elif DEVICE.type == "mps":
                torch.mps.empty_cache()

        train_loss = running_loss / max(total, 1)
        train_acc = correct / max(total, 1)
        val_acc, _, _, _ = evaluate_loader(model, val_loader)

        print(f"Epoch {epoch + 1:02d} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), FUSION_MODEL_PATH)

    print(f"\nBest concat fusion model saved to: {FUSION_MODEL_PATH}")
    return model


def evaluate_test(model: ConcatFusionHead, df: pd.DataFrame) -> None:
    _, _, test_loader = make_loaders(df)
    acc, y_true, y_pred, ids = evaluate_loader(model, test_loader)

    print("\n" + "=" * 80)
    print("CONCAT FUSION TEST RESULTS")
    print(f"Test Accuracy: {acc:.4f}")
    print(classification_report(y_true, y_pred, target_names=[CLASS_MAP_STR[i] for i in range(NUM_CLASSES)]))

    results = pd.DataFrame({
        "BDD_ID": ids,
        "true_idx": y_true,
        "pred_idx": y_pred,
        "true_label": [CLASS_MAP_STR[i] for i in y_true],
        "pred_label": [CLASS_MAP_STR[i] for i in y_pred],
    })
    results.to_csv(TEST_PRED_PATH, index=False)
    print(f"Saved concat-fusion test predictions to: {TEST_PRED_PATH}")
    print("=" * 80)


def main():
    set_seed(SEED)
    df = build_fusion_dataframe()
    model = train_concat_fusion(df)
    model.load_state_dict(torch.load(FUSION_MODEL_PATH, map_location=DEVICE, weights_only=False))
    evaluate_test(model, df)


if __name__ == "__main__":
    main()
