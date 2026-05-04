
import json
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

# =========================================================
# CONFIG
# =========================================================
SLOWFAST_SPLITS_PATH = Path("data/slowfast_video_only/video_only_splits.csv")

# notebook artifacts
X_TS_PATH = Path("X_ts.npy")
META_PATH = Path("meta.csv")
X_CTX_PATH = Path("X_ctx.csv")

OUT_DIR = Path("data/slowfast_joint_fusion")
OUT_DIR.mkdir(parents=True, exist_ok=True)
MERGED_SPLITS_PATH = OUT_DIR / "joint_fusion_splits.csv"
MODEL_PATH = OUT_DIR / "joint_fusion_best.pth"
TEST_PRED_PATH = OUT_DIR / "joint_fusion_test_predictions.csv"

SEED = 1
NUM_CLASSES = 4
BATCH_SIZE = 4
EPOCHS = 20
PATIENCE = 6
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 1e-4

# Fine-tuning controls
UNFREEZE_VIDEO_BLOCKS = ["blocks.5"]   # last residual stage
FREEZE_KINE_BRANCH = False             # set True if you want only fusion + video tuning
CLASS_MAP_STR = {
    0: "Conflict",
    1: "Bump",
    2: "Hard Brake",
    3: "Not an SCE",
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

        return torch.cat((x_ts_features, x_ctx_features), dim=1)  # 164-d

    def forward(self, x_ts: torch.Tensor, x_ctx: torch.Tensor) -> torch.Tensor:
        fused = self.extract_fused_features(x_ts, x_ctx)
        out = self.fusion_drop(fused)
        out = self.fc_final(out)
        return out

# =========================================================
# TRUE JOINT MODEL
# =========================================================
class JointSlowFastKineFusion(nn.Module):
    """
    True joint model:
    - SlowFast runs during training on saved transformed tensors
    - CNN/LSTM runs during training on kinematics/context
    - fuse BOTH branches before final layers
    - gradients flow through fusion head + kine branch + selected SlowFast blocks
    """
    def __init__(self, in_chans_ts: int, in_chans_ctx: int, num_classes: int = NUM_CLASSES) -> None:
        super().__init__()

        self.video_backbone = slowfast_r50(pretrained=True)
        self.video_backbone.blocks[6].proj = nn.Identity()  # 2304-d embedding

        self.kine_branch = MultiModal_CNN_LSTM(
            in_chans_ts=in_chans_ts,
            in_chans_ctx=in_chans_ctx,
            num_classes=num_classes,
        )

        self.video_proj = nn.Sequential(
            nn.Linear(2304, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
        )
        self.kine_proj = nn.Sequential(
            nn.Linear(164, 256),
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
        self.classifier = nn.Linear(128, num_classes)

    def freeze_for_finetune(self) -> None:
        # Freeze all SlowFast params first
        for p in self.video_backbone.parameters():
            p.requires_grad = False

        # Unfreeze selected later blocks
        for name, p in self.video_backbone.named_parameters():
            if any(tag in name for tag in UNFREEZE_VIDEO_BLOCKS):
                p.requires_grad = True

        # Kine branch can be jointly tuned or frozen
        for p in self.kine_branch.parameters():
            p.requires_grad = not FREEZE_KINE_BRANCH

        # Fusion layers always train
        for module in [self.video_proj, self.kine_proj, self.fusion, self.classifier]:
            for p in module.parameters():
                p.requires_grad = True

    def forward(
        self,
        slow_frames: torch.Tensor,
        fast_frames: torch.Tensor,
        x_ts: torch.Tensor,
        x_ctx: torch.Tensor,
    ) -> torch.Tensor:
        video_emb = self.video_backbone([slow_frames, fast_frames])
        kine_emb = self.kine_branch.extract_fused_features(x_ts, x_ctx)

        v = self.video_proj(video_emb)
        k = self.kine_proj(kine_emb)
        joint = torch.cat([v, k, v * k, torch.abs(v - k)], dim=1)
        fused = self.fusion(joint)
        return self.classifier(fused)

# =========================================================
# DATA
# =========================================================
class JointFusionDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, X_ts: np.ndarray, X_ctx: np.ndarray) -> None:
        self.df = dataframe.reset_index(drop=True)
        self.X_ts = X_ts
        self.X_ctx = X_ctx

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        tensor_blob = torch.load(row["tensor_path"], map_location="cpu", weights_only=False)
        slow = tensor_blob["slow"].float()
        fast = tensor_blob["fast"].float()

        row_idx = int(row["row_idx"])
        x_ts = torch.tensor(self.X_ts[row_idx], dtype=torch.float32)
        x_ctx = torch.tensor(self.X_ctx[row_idx], dtype=torch.float32)

        target = int(row["target_idx"])
        bdd_id = str(row["BDD_ID"])
        return slow, fast, x_ts, x_ctx, target, bdd_id

def load_and_align_data() -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    if not SLOWFAST_SPLITS_PATH.exists():
        raise FileNotFoundError(
            f"Missing {SLOWFAST_SPLITS_PATH}. Run the video-only SlowFast pipeline first so tensor_path exists."
        )

    slowfast_df = pd.read_csv(SLOWFAST_SPLITS_PATH).copy()
    if "tensor_path" not in slowfast_df.columns:
        raise ValueError("video_only_splits.csv must contain tensor_path. Re-run slowfast_video_only if needed.")

    meta = pd.read_csv(META_PATH).copy()
    ctx = pd.read_csv(X_CTX_PATH).copy()
    X_ts = np.load(X_TS_PATH).astype(np.float32)

    if "BDD_ID" not in meta.columns:
        raise ValueError("meta.csv must contain BDD_ID.")
    if "y" not in meta.columns:
        raise ValueError("meta.csv must contain zero-indexed y labels in column y.")

    meta["BDD_ID"] = meta["BDD_ID"].astype(str)
    meta["row_idx"] = np.arange(len(meta))

    slowfast_df["BDD_ID"] = slowfast_df["BDD_ID"].astype(str)
    merged = slowfast_df.merge(meta[["BDD_ID", "row_idx", "y"]], on="BDD_ID", how="inner")
    merged = merged.drop_duplicates(subset=["BDD_ID"]).reset_index(drop=True)

    # Use the video split/target as canonical if present
    if "target_idx" not in merged.columns:
        merged["target_idx"] = merged["y"].astype(int)

    # Rebuild context exactly like notebook-style scripts
    ctx_features = ctx.drop(columns=[c for c in ["BDD_ID", "EVENT_ID", "EVENT_TYPE", "y"] if c in ctx.columns], errors="ignore")
    categorical_cols = [c for c in ["weather", "scene", "timeofday"] if c in ctx_features.columns]
    ctx_features = pd.get_dummies(ctx_features, columns=categorical_cols, dummy_na=True)
    ctx_features = ctx_features.fillna(0)
    X_ctx_all = ctx_features.to_numpy(dtype=np.float32)

    train_mask = merged["split"].to_numpy() == "train"
    row_idx = merged["row_idx"].to_numpy(dtype=int)

    X_ctx_sel = X_ctx_all.copy()
    scaler = StandardScaler()
    X_ctx_sel[row_idx[train_mask]] = scaler.fit_transform(X_ctx_sel[row_idx[train_mask]])
    X_ctx_sel[row_idx[~train_mask]] = scaler.transform(X_ctx_sel[row_idx[~train_mask]])

    merged.to_csv(MERGED_SPLITS_PATH, index=False)
    print(f"Saved joint-fusion metadata to: {MERGED_SPLITS_PATH}")
    return merged, X_ts, X_ctx_sel.astype(np.float32)

# =========================================================
# TRAIN / EVAL
# =========================================================
def make_loaders(df: pd.DataFrame, X_ts: np.ndarray, X_ctx: np.ndarray):
    train_df = df[df["split"] == "train"].copy()
    val_df = df[df["split"] == "val"].copy()
    test_df = df[df["split"] == "test"].copy()

    train_loader = DataLoader(JointFusionDataset(train_df, X_ts, X_ctx), batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=False)
    val_loader = DataLoader(JointFusionDataset(val_df, X_ts, X_ctx), batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=False)
    test_loader = DataLoader(JointFusionDataset(test_df, X_ts, X_ctx), batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=False)
    return train_loader, val_loader, test_loader

def evaluate_loader(model: nn.Module, loader: DataLoader):
    model.eval()
    y_true, y_pred, ids = [], [], []

    with torch.no_grad():
        for slow, fast, x_ts, x_ctx, targets, batch_ids in loader:
            slow = slow.to(DEVICE)
            fast = fast.to(DEVICE)
            x_ts = x_ts.to(DEVICE)
            x_ctx = x_ctx.to(DEVICE)
            logits = model(slow, fast, x_ts, x_ctx)
            preds = torch.argmax(logits, dim=1).cpu().numpy()

            y_true.extend(targets.numpy().tolist())
            y_pred.extend(preds.tolist())
            ids.extend(list(batch_ids))

    return accuracy_score(y_true, y_pred), y_true, y_pred, ids

def train_joint_model(df: pd.DataFrame, X_ts: np.ndarray, X_ctx: np.ndarray) -> JointSlowFastKineFusion:
    train_loader, val_loader, _ = make_loaders(df, X_ts, X_ctx)

    in_chans_ts = X_ts.shape[1]
    in_chans_ctx = X_ctx.shape[1]
    model = JointSlowFastKineFusion(in_chans_ts=in_chans_ts, in_chans_ctx=in_chans_ctx).to(DEVICE)
    model.freeze_for_finetune()

    train_targets = df[df["split"] == "train"]["target_idx"].to_numpy(dtype=int)
    class_counts = np.bincount(train_targets, minlength=NUM_CLASSES)
    class_weights = len(train_targets) / (NUM_CLASSES * np.maximum(class_counts, 1))
    class_weights = torch.tensor(class_weights, dtype=torch.float32, device=DEVICE)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    best_val_acc = -1.0
    patience_ctr = 0
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        total = 0
        correct = 0

        for slow, fast, x_ts, x_ctx, targets, _ in tqdm(train_loader, desc=f"Joint Epoch {epoch + 1:02d}/{EPOCHS}", leave=False):
            slow = slow.to(DEVICE)
            fast = fast.to(DEVICE)
            x_ts = x_ts.to(DEVICE)
            x_ctx = x_ctx.to(DEVICE)
            targets = targets.to(DEVICE)

            optimizer.zero_grad()
            logits = model(slow, fast, x_ts, x_ctx)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * targets.size(0)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)

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

    print(f"\nBest joint model saved to: {MODEL_PATH}")
    return model

def evaluate_test(model: JointSlowFastKineFusion, df: pd.DataFrame, X_ts: np.ndarray, X_ctx: np.ndarray) -> None:
    _, _, test_loader = make_loaders(df, X_ts, X_ctx)
    acc, y_true, y_pred, ids = evaluate_loader(model, test_loader)

    print("\n" + "=" * 80)
    print("TRUE JOINT FUSION TEST RESULTS")
    print(f"Test Accuracy: {acc:.4f}")
    print(classification_report(y_true, y_pred, target_names=[CLASS_MAP_STR[i] for i in range(NUM_CLASSES)]))

    out = pd.DataFrame({
        "BDD_ID": ids,
        "true_idx": y_true,
        "pred_idx": y_pred,
        "true_label": [CLASS_MAP_STR[i] for i in y_true],
        "pred_label": [CLASS_MAP_STR[i] for i in y_pred],
    })
    out.to_csv(TEST_PRED_PATH, index=False)
    print(f"Saved joint-fusion test predictions to: {TEST_PRED_PATH}")
    print("=" * 80)

def main():
    set_seed(SEED)
    df, X_ts, X_ctx = load_and_align_data()
    model = train_joint_model(df, X_ts, X_ctx)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False))
    evaluate_test(model, df, X_ts, X_ctx)

if __name__ == "__main__":
    main()
