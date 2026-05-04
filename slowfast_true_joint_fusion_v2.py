import sys
from types import ModuleType
try:
    import torchvision.transforms.functional_tensor  # type: ignore
except ImportError:
    import torchvision.transforms.functional as F_base
    fake_module = ModuleType("torchvision.transforms.functional_tensor")
    for attr in dir(F_base):
        setattr(fake_module, attr, getattr(F_base, attr))
    sys.modules["torchvision.transforms.functional_tensor"] = fake_module

from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from pytorchvideo.models.hub import slowfast_r50

VIDEO_SPLITS_PATH = Path("data/slowfast_video_only/video_only_splits.csv")
KINE_X_PATH = Path("data/X_ts.npy")
KINE_META_PATH = Path("data/meta.csv")
KINE_CTX_PATH = Path("data/X_ctx.csv")

OUT_DIR = Path("data/slowfast_joint_fusion_v2")
OUT_DIR.mkdir(parents=True, exist_ok=True)
JOINT_META_PATH = OUT_DIR / "joint_fusion_splits.csv"
MODEL_PATH = OUT_DIR / "joint_fusion_v2_best.pth"
PRED_PATH = OUT_DIR / "joint_fusion_v2_test_predictions.csv"

SEED = 1
NUM_CLASSES = 4
BATCH_SIZE = 4
EPOCHS = 24
PATIENCE = 5
WEIGHT_DECAY = 1e-4

LR_FUSION = 1e-3
LR_KINE = 2e-4
LR_SLOWFAST = 2e-5

TRAIN_SLOWFAST = True
UNFREEZE_BLOCK5 = False
UNFREEZE_PROJ_HEAD = False
FREEZE_KINE_BRANCH = True

CLASS_MAP_STR = {
    0: "Conflict",
    1: "Bump",
    2: "Hard Brake",
    3: "Not an SCE",
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {DEVICE.type.upper()} device!")


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class MultiModal_CNN_LSTM_Feature(nn.Module):
    def __init__(self, in_chans_ts: int, in_chans_ctx: int, num_classes: int) -> None:
        super().__init__()
        self.bn_input = nn.BatchNorm1d(in_chans_ts)
        self.conv1 = nn.Conv1d(in_channels=in_chans_ts, out_channels=64, kernel_size=12, stride=1, padding="same")
        self.bn_conv = nn.BatchNorm1d(64)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.lstm = nn.LSTM(input_size=64, hidden_size=100, num_layers=2, batch_first=True, dropout=0.3)

        self.ctx_fc1 = nn.Linear(in_chans_ctx, 64)
        self.ctx_norm = nn.LayerNorm(64)
        self.ctx_relu = nn.ReLU()
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
        x_ctx_features = self.ctx_norm(x_ctx_features)
        x_ctx_features = self.ctx_relu(x_ctx_features)
        x_ctx_features = self.ctx_drop(x_ctx_features)

        return torch.cat((x_ts_features, x_ctx_features), dim=1)

    def forward(self, x_ts: torch.Tensor, x_ctx: torch.Tensor) -> torch.Tensor:
        fused = self.extract_fused_features(x_ts, x_ctx)
        out = self.fusion_drop(fused)
        return self.fc_final(out)


class SlowFastFinetuneBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = slowfast_r50(pretrained=True)
        self.backbone.blocks[6].proj = nn.Identity()

    def forward(self, video_pathway: List[torch.Tensor]) -> torch.Tensor:
        return self.backbone(video_pathway)


class SaferJointFusionModel(nn.Module):
    def __init__(self, in_chans_ts: int, in_chans_ctx: int, num_classes: int = NUM_CLASSES) -> None:
        super().__init__()
        self.video_backbone = SlowFastFinetuneBackbone()
        self.kine_branch = MultiModal_CNN_LSTM_Feature(in_chans_ts, in_chans_ctx, num_classes)

        self.video_proj = nn.Sequential(
            nn.Linear(2304, 512),
            nn.ReLU(),
            nn.Dropout(0.35),
            nn.Linear(512, 256),
            nn.ReLU(),
        )
        self.kine_proj = nn.Sequential(
            nn.Linear(164, 256),
            nn.ReLU(),
            nn.Dropout(0.25),
        )
        self.fusion = nn.Sequential(
            nn.Linear(256 * 4, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        self.classifier = nn.Linear(128, num_classes)
        self._apply_freezing()

    def _apply_freezing(self) -> None:
        for p in self.video_backbone.parameters():
            p.requires_grad = False

        if TRAIN_SLOWFAST:
            for name, p in self.video_backbone.named_parameters():
                if UNFREEZE_BLOCK5 and "backbone.blocks.5" in name:
                    p.requires_grad = True
                if UNFREEZE_PROJ_HEAD and "backbone.blocks.6" in name:
                    p.requires_grad = True

        if FREEZE_KINE_BRANCH:
            for p in self.kine_branch.parameters():
                p.requires_grad = False

    def forward(self, slow: torch.Tensor, fast: torch.Tensor, x_ts: torch.Tensor, x_ctx: torch.Tensor) -> torch.Tensor:
        vid_emb = self.video_backbone([slow, fast])
        kine_emb = self.kine_branch.extract_fused_features(x_ts, x_ctx)

        v = self.video_proj(vid_emb)
        k = self.kine_proj(kine_emb)
        joint = torch.cat([v, k, v * k, torch.abs(v - k)], dim=1)
        fused = self.fusion(joint)
        return self.classifier(fused)


class JointTensorDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, X_ts: np.ndarray, X_ctx: np.ndarray) -> None:
        self.df = dataframe.reset_index(drop=True)
        self.X_ts = X_ts
        self.X_ctx = X_ctx

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        saved = torch.load(row["tensor_path"], map_location="cpu", weights_only=False)
        slow = saved["slow"].float()
        fast = saved["fast"].float()
        aligned_idx = int(row["aligned_idx"])
        x_ts = torch.tensor(self.X_ts[aligned_idx], dtype=torch.float32)
        x_ctx = torch.tensor(self.X_ctx[aligned_idx], dtype=torch.float32)
        target = int(row["target_idx"])
        bdd_id = str(row["BDD_ID"])
        return slow, fast, x_ts, x_ctx, target, bdd_id


def load_and_align_data() -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    if not VIDEO_SPLITS_PATH.exists():
        raise FileNotFoundError(f"Missing {VIDEO_SPLITS_PATH}. Run the video-only SlowFast pipeline first.")

    df = pd.read_csv(VIDEO_SPLITS_PATH).copy()
    df["BDD_ID"] = df["BDD_ID"].astype(str)

    meta = pd.read_csv(KINE_META_PATH).copy()
    ctx = pd.read_csv(KINE_CTX_PATH).copy()
    X_ts = np.load(KINE_X_PATH).astype(np.float32)

    assert len(meta) == len(ctx) == len(X_ts), "Mismatch among meta, ctx, and X_ts lengths"
    assert "BDD_ID" in meta.columns, "meta.csv must contain BDD_ID"

    meta["BDD_ID"] = meta["BDD_ID"].astype(str)
    meta["row_idx"] = np.arange(len(meta))

    merged = df.merge(meta[["BDD_ID", "row_idx"]], on="BDD_ID", how="inner")
    merged = merged.drop_duplicates(subset=["BDD_ID"]).reset_index(drop=True)

    ctx_features = ctx.drop(columns=[c for c in ["BDD_ID", "EVENT_ID", "EVENT_TYPE", "y"] if c in ctx.columns], errors="ignore")
    categorical_cols = [c for c in ["weather", "scene", "timeofday"] if c in ctx_features.columns]
    ctx_features = pd.get_dummies(ctx_features, columns=categorical_cols, dummy_na=True)
    ctx_features = ctx_features.fillna(0)
    X_ctx_all = ctx_features.to_numpy(dtype=np.float32)

    row_idx = merged["row_idx"].to_numpy(dtype=int)

    # Filter BOTH arrays to the merged rows, in the same order.
    X_ts_sel = X_ts[row_idx].astype(np.float32)
    X_ctx_sel = X_ctx_all[row_idx].astype(np.float32)

    # Scale context features using only the train split.
    train_mask = merged["split"].to_numpy() == "train"
    scaler = StandardScaler()
    X_ctx_sel[train_mask] = scaler.fit_transform(X_ctx_sel[train_mask])
    X_ctx_sel[~train_mask] = scaler.transform(X_ctx_sel[~train_mask])

    # Create a fresh aligned index for the filtered arrays.
    merged = merged.reset_index(drop=True)
    merged["aligned_idx"] = np.arange(len(merged))

    merged.to_csv(JOINT_META_PATH, index=False)
    print(f"Saved joint-fusion metadata to: {JOINT_META_PATH}")
    return merged, X_ts_sel, X_ctx_sel


def make_loaders(df: pd.DataFrame, X_ts: np.ndarray, X_ctx: np.ndarray):
    train_df = df[df["split"] == "train"].copy()
    val_df = df[df["split"] == "val"].copy()
    test_df = df[df["split"] == "test"].copy()

    train_loader = DataLoader(
        JointTensorDataset(train_df, X_ts, X_ctx),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        drop_last=True,
    )
    val_loader = DataLoader(
        JointTensorDataset(val_df, X_ts, X_ctx),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )
    test_loader = DataLoader(
        JointTensorDataset(test_df, X_ts, X_ctx),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )
    return train_loader, val_loader, test_loader


def build_optimizer(model: SaferJointFusionModel) -> torch.optim.Optimizer:
    param_groups = []

    fusion_params = list(model.video_proj.parameters()) + list(model.kine_proj.parameters()) + list(model.fusion.parameters()) + list(model.classifier.parameters())
    fusion_params = [p for p in fusion_params if p.requires_grad]
    if fusion_params:
        param_groups.append({"params": fusion_params, "lr": LR_FUSION})

    kine_params = [p for p in model.kine_branch.parameters() if p.requires_grad]
    if kine_params:
        param_groups.append({"params": kine_params, "lr": LR_KINE})

    slowfast_params = [p for p in model.video_backbone.parameters() if p.requires_grad]
    if slowfast_params:
        param_groups.append({"params": slowfast_params, "lr": LR_SLOWFAST})

    return torch.optim.AdamW(param_groups, weight_decay=WEIGHT_DECAY)


def evaluate_loader(model: nn.Module, loader: DataLoader) -> Tuple[float, List[int], List[int], List[str]]:
    model.eval()
    y_true: List[int] = []
    y_pred: List[int] = []
    ids: List[str] = []

    with torch.no_grad():
        for slow, fast, x_ts, x_ctx, batch_targets, batch_ids in loader:
            slow = slow.to(DEVICE)
            fast = fast.to(DEVICE)
            x_ts = x_ts.to(DEVICE)
            x_ctx = x_ctx.to(DEVICE)

            logits = model(slow, fast, x_ts, x_ctx)
            preds = torch.argmax(logits, dim=1).cpu().numpy()

            y_true.extend(batch_targets.numpy().tolist())
            y_pred.extend(preds.tolist())
            ids.extend(list(batch_ids))

    acc = accuracy_score(y_true, y_pred)
    return acc, y_true, y_pred, ids


def train_joint_model(df: pd.DataFrame, X_ts: np.ndarray, X_ctx: np.ndarray) -> SaferJointFusionModel:
    train_loader, val_loader, _ = make_loaders(df, X_ts, X_ctx)

    model = SaferJointFusionModel(in_chans_ts=X_ts.shape[1], in_chans_ctx=X_ctx.shape[1]).to(DEVICE)

    train_targets = df[df["split"] == "train"]["target_idx"].to_numpy(dtype=int)
    class_counts = np.bincount(train_targets, minlength=NUM_CLASSES)
    class_weights = len(train_targets) / (NUM_CLASSES * np.maximum(class_counts, 1))
    class_weights = torch.tensor(class_weights, dtype=torch.float32, device=DEVICE)

    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)
    optimizer = build_optimizer(model)

    best_val_acc = -1.0
    patience_ctr = 0
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        total = 0
        correct = 0

        for slow, fast, x_ts, x_ctx, batch_targets, _ in tqdm(train_loader, desc=f"JointV2 Epoch {epoch + 1:02d}/{EPOCHS}", leave=False):
            slow = slow.to(DEVICE)
            fast = fast.to(DEVICE)
            x_ts = x_ts.to(DEVICE)
            x_ctx = x_ctx.to(DEVICE)
            batch_targets = batch_targets.to(DEVICE)

            optimizer.zero_grad()
            logits = model(slow, fast, x_ts, x_ctx)
            loss = criterion(logits, batch_targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item() * batch_targets.size(0)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == batch_targets).sum().item()
            total += batch_targets.size(0)

            del logits, loss, preds
            if DEVICE.type == "cuda":
                torch.cuda.empty_cache()

        train_loss = running_loss / max(total, 1)
        train_acc = correct / max(total, 1)
        val_acc, _, _, _ = evaluate_loader(model, val_loader)
        print(f"Epoch {epoch + 1:02d} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_ctr = 0
            torch.save(model.state_dict(), MODEL_PATH)
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                print(f"Early stopping at epoch {epoch + 1}.")
                break

    print(f"\nBest joint v2 model saved to: {MODEL_PATH}")
    return model


def evaluate_test(model: SaferJointFusionModel, df: pd.DataFrame, X_ts: np.ndarray, X_ctx: np.ndarray) -> None:
    _, _, test_loader = make_loaders(df, X_ts, X_ctx)
    acc, y_true, y_pred, ids = evaluate_loader(model, test_loader)

    print("\n" + "=" * 80)
    print("TRUE JOINT FUSION V2 TEST RESULTS")
    print(f"Test Accuracy: {acc:.4f}")
    print(classification_report(y_true, y_pred, target_names=[CLASS_MAP_STR[i] for i in range(NUM_CLASSES)]))

    out = pd.DataFrame({
        "BDD_ID": ids,
        "true_idx": y_true,
        "pred_idx": y_pred,
        "true_label": [CLASS_MAP_STR[i] for i in y_true],
        "pred_label": [CLASS_MAP_STR[i] for i in y_pred],
    })
    out.to_csv(PRED_PATH, index=False)
    print(f"Saved joint v2 test predictions to: {PRED_PATH}")
    print("=" * 80)


def main() -> None:
    set_seed(SEED)
    df, X_ts, X_ctx = load_and_align_data()
    model = train_joint_model(df, X_ts, X_ctx)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False))
    evaluate_test(model, df, X_ts, X_ctx)


if __name__ == "__main__":
    main()
