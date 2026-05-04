import json
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.config import Config
from mmaction.apis import init_recognizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# CONFIG
VIDEO_SPLITS_PATH = Path("data/slowfast_video_only/video_only_splits.csv")
KINE_X_PATH = Path("data/X_ts.npy")
KINE_META_PATH = Path("data/meta.csv")
KINE_CTX_PATH = Path("data/X_ctx.csv")
KINE_MODEL_PATH = Path("best_model.pth")

# Update these two to match your local MMAction2 install.
MMACTION_CONFIG_PATH = Path("configs/recognition/slowfast/slowfast_r50_8xb8-4x16x1-256e_kinetics400-rgb.py")
MMACTION_CHECKPOINT_PATH = Path("checkpoints/slowfast_r50_kinetics400.pth")

# If VIDEO_SPLITS_PATH does not already contain a full path column, set VIDEO_ROOT.
VIDEO_ROOT = Path("data/videos")

CACHE_DIR = Path("data/mmaction2_slowfast_midlevel_fusion")
VIDEO_EMBED_DIR = CACHE_DIR / "video_embeddings"
KINE_EMBED_DIR = CACHE_DIR / "kine_embeddings"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
VIDEO_EMBED_DIR.mkdir(parents=True, exist_ok=True)
KINE_EMBED_DIR.mkdir(parents=True, exist_ok=True)

FUSION_META_PATH = CACHE_DIR / "mmaction2_midlevel_fusion_splits.csv"
MODEL_PATH = CACHE_DIR / "mmaction2_midlevel_fusion_best.pth"
PRED_PATH = CACHE_DIR / "mmaction2_midlevel_fusion_test_predictions.csv"

BATCH_SIZE = 64
EPOCHS = 40
PATIENCE = 10
LEARNING_RATE = 1e-3
SEED = 1
NUM_CLASSES = 4

CLIP_LEN = 32
FRAME_INTERVAL = 2
INPUT_SIZE = 224
RESIZE_SHORT_SIDE = 256

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

        return torch.cat((x_ts_features, x_ctx_features), dim=1)

    def forward(self, x_ts: torch.Tensor, x_ctx: torch.Tensor) -> torch.Tensor:
        fused = self.extract_fused_features(x_ts, x_ctx)
        return self.fc_final(self.fusion_drop(fused))


# =========================================================
# MMAction2 SlowFast video feature extraction
# =========================================================
class MMAction2SlowFastExtractor:
    """
    Uses an MMAction2 SlowFast recognizer and exports pooled backbone features.
    For SlowFast-R50 this is typically 2048 (slow) + 256 (fast) = 2304 dims.
    """
    def __init__(self, config_path: Path, checkpoint_path: Path, device: torch.device) -> None:
        if not config_path.exists():
            raise FileNotFoundError(f"MMAction2 config not found: {config_path}")
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"MMAction2 checkpoint not found: {checkpoint_path}")

        self.cfg = Config.fromfile(str(config_path))
        self.model = init_recognizer(str(config_path), str(checkpoint_path), device=str(device))
        self.model.eval()
        self.device = device

        dp = self.cfg.model.get("data_preprocessor", {})
        mean = dp.get("mean", [123.675, 116.28, 103.53])
        std = dp.get("std", [58.395, 57.12, 57.375])
        self.mean = torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1, 1)

    @staticmethod
    def _resize_short_side(frame: np.ndarray, short_side: int) -> np.ndarray:
        h, w = frame.shape[:2]
        if h < w:
            new_h = short_side
            new_w = int(w * short_side / h)
        else:
            new_w = short_side
            new_h = int(h * short_side / w)
        return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    @staticmethod
    def _center_crop(frame: np.ndarray, crop_size: int) -> np.ndarray:
        h, w = frame.shape[:2]
        top = max((h - crop_size) // 2, 0)
        left = max((w - crop_size) // 2, 0)
        return frame[top:top + crop_size, left:left + crop_size]

    def _read_video_frames(self, video_path: Path) -> List[np.ndarray]:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")

        frames: List[np.ndarray] = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()

        if not frames:
            raise RuntimeError(f"No frames decoded from: {video_path}")
        return frames

    def _sample_clip_indices(self, num_frames: int) -> np.ndarray:
        required = CLIP_LEN * FRAME_INTERVAL
        if num_frames >= required:
            center = num_frames // 2
            start = max(0, center - required // 2)
            end = min(num_frames, start + required)
            start = max(0, end - required)
            idxs = np.arange(start, end, FRAME_INTERVAL)
        else:
            idxs = np.linspace(0, max(num_frames - 1, 0), CLIP_LEN).astype(int)

        if len(idxs) < CLIP_LEN:
            pad_val = idxs[-1] if len(idxs) > 0 else 0
            idxs = np.pad(idxs, (0, CLIP_LEN - len(idxs)), constant_values=pad_val)
        return idxs[:CLIP_LEN]

    def _frames_to_tensor(self, frames: List[np.ndarray]) -> torch.Tensor:
        idxs = self._sample_clip_indices(len(frames))
        clip = []
        for idx in idxs:
            frame = frames[int(idx)]
            frame = self._resize_short_side(frame, RESIZE_SHORT_SIDE)
            frame = self._center_crop(frame, INPUT_SIZE)
            if frame.shape[:2] != (INPUT_SIZE, INPUT_SIZE):
                frame = cv2.resize(frame, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_LINEAR)
            clip.append(frame)

        clip_np = np.stack(clip, axis=0).astype(np.float32)   # [T, H, W, C]
        clip_np = clip_np.transpose(3, 0, 1, 2)              # [C, T, H, W]
        tensor = torch.from_numpy(clip_np).unsqueeze(0)      # [1, C, T, H, W]
        return (tensor - self.mean) / self.std

    @torch.no_grad()
    def extract_embedding(self, video_path: Path) -> torch.Tensor:
        frames = self._read_video_frames(video_path)
        video_tensor = self._frames_to_tensor(frames).to(self.device)

        feats = self.model.backbone(video_tensor)
        if isinstance(feats, tuple):
            pooled = []
            for feat in feats:
                pooled.append(F.adaptive_avg_pool3d(feat, (1, 1, 1)).flatten(1))
            emb = torch.cat(pooled, dim=1)
        elif isinstance(feats, torch.Tensor):
            emb = F.adaptive_avg_pool3d(feats, (1, 1, 1)).flatten(1)
        else:
            raise TypeError(f"Unexpected backbone output type: {type(feats)}")

        return emb.squeeze(0).detach().cpu()


# =========================================================
# MID-LEVEL FUSION MODEL
# =========================================================
class MidLevelFusionModel(nn.Module):
    def __init__(self, video_dim: int = 2304, kine_dim: int = 164, num_classes: int = NUM_CLASSES) -> None:
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
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, video_embedding: torch.Tensor, kine_embedding: torch.Tensor) -> torch.Tensor:
        v = self.video_proj(video_embedding)
        k = self.kine_proj(kine_embedding)
        joint = torch.cat([v, k, v * k, torch.abs(v - k)], dim=1)
        fused = self.fusion(joint)
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
    def __init__(self, dataframe: pd.DataFrame) -> None:
        self.df = dataframe.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        video_emb = torch.load(row["embedding_path"], map_location="cpu", weights_only=False).float()
        kine_emb = torch.load(row["kine_embedding_path"], map_location="cpu", weights_only=False).float()
        target = int(row["target_idx"])
        bdd_id = str(row["BDD_ID"])
        return video_emb, kine_emb, target, bdd_id


# =========================================================
# PREP / MERGE
# =========================================================
def infer_video_path(row: pd.Series) -> Path:
    for col in ["video_path", "video_file", "filename", "file_name", "clip_path"]:
        if col in row.index and pd.notna(row[col]):
            path = Path(str(row[col]))
            return path if path.is_absolute() else (VIDEO_ROOT / path)

    bdd_id = str(row["BDD_ID"])
    for ext in [".mp4", ".mov", ".avi", ".mkv"]:
        candidate = VIDEO_ROOT / f"{bdd_id}{ext}"
        if candidate.exists():
            return candidate

    raise KeyError(
        "Could not infer a video path. Add one of these columns to VIDEO_SPLITS_PATH: "
        "video_path, video_file, filename, file_name, clip_path."
    )


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

    merged["video_path"] = merged.apply(infer_video_path, axis=1).astype(str)

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

    return merged, X_ts_sel, X_ctx_sel


@torch.no_grad()
def export_video_embeddings_mmaction2(df: pd.DataFrame) -> pd.DataFrame:
    print("Exporting MMAction2 SlowFast video embeddings...")
    extractor = MMAction2SlowFastExtractor(MMACTION_CONFIG_PATH, MMACTION_CHECKPOINT_PATH, DEVICE)

    embed_map = {}
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Exporting MMAction2 SlowFast embeddings"):
        bdd_id = str(row["BDD_ID"])
        video_path = Path(str(row["video_path"]))
        out_path = VIDEO_EMBED_DIR / f"{bdd_id}.pt"

        if not out_path.exists():
            emb = extractor.extract_embedding(video_path)
            torch.save(emb, out_path)
        embed_map[bdd_id] = str(out_path)

    out_df = df.copy()
    out_df["embedding_path"] = out_df["BDD_ID"].map(embed_map)
    return out_df.dropna(subset=["embedding_path"]).reset_index(drop=True)


@torch.no_grad()
def export_kine_embeddings(df: pd.DataFrame, X_ts: np.ndarray, X_ctx: np.ndarray) -> pd.DataFrame:
    if not KINE_MODEL_PATH.exists():
        raise FileNotFoundError(f"Could not find {KINE_MODEL_PATH}. Rerun the notebook first so best_model.pth exists.")

    print("Exporting kinematics/context embeddings...")
    ds = KineFeatureDataset(X_ts, X_ctx, df["BDD_ID"].to_numpy())
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=False)

    model = MultiModal_CNN_LSTM(
        in_chans_ts=X_ts.shape[1],
        in_chans_ctx=X_ctx.shape[1],
        num_classes=NUM_CLASSES,
    ).to(DEVICE)
    model.load_state_dict(torch.load(KINE_MODEL_PATH, map_location=DEVICE, weights_only=False))
    model.eval()

    embed_map = {}
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
    print(f"Saved MMAction2 mid-level fusion metadata to: {FUSION_META_PATH}")
    return out_df


# =========================================================
# TRAIN / EVAL
# =========================================================
def make_loaders(df: pd.DataFrame) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_df = df[df["split"] == "train"].copy()
    val_df = df[df["split"] == "val"].copy()
    test_df = df[df["split"] == "test"].copy()

    train_loader = DataLoader(MidLevelFusionDataset(train_df), batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=False)
    val_loader = DataLoader(MidLevelFusionDataset(val_df), batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=False)
    test_loader = DataLoader(MidLevelFusionDataset(test_df), batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=False)
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

    acc = accuracy_score(y_true, y_pred)
    return acc, y_true, y_pred, ids


def infer_video_dim(df: pd.DataFrame) -> int:
    example_path = df["embedding_path"].dropna().iloc[0]
    emb = torch.load(example_path, map_location="cpu", weights_only=False).float()
    return int(emb.numel())


def train_model(df: pd.DataFrame) -> MidLevelFusionModel:
    train_loader, val_loader, _ = make_loaders(df)
    video_dim = infer_video_dim(df)
    model = MidLevelFusionModel(video_dim=video_dim).to(DEVICE)

    train_targets = df[df["split"] == "train"]["target_idx"].to_numpy(dtype=int)
    class_counts = np.bincount(train_targets, minlength=NUM_CLASSES)
    class_weights = len(train_targets) / (NUM_CLASSES * np.maximum(class_counts, 1))
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

        for batch_v, batch_k, batch_targets, _ in tqdm(train_loader, desc=f"Epoch {epoch + 1:02d}/{EPOCHS}", leave=False):
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

    print(f"\nBest MMAction2 mid-level fusion model saved to: {MODEL_PATH}")
    return model


def evaluate_test(model: MidLevelFusionModel, df: pd.DataFrame) -> None:
    _, _, test_loader = make_loaders(df)
    acc, y_true, y_pred, ids = evaluate_loader(model, test_loader)

    print("\n" + "=" * 80)
    print("MMACTION2 SLOWFAST MID-LEVEL FUSION TEST RESULTS")
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
    print(f"Saved MMAction2 SlowFast mid-level fusion test predictions to: {PRED_PATH}")
    print("=" * 80)


# =========================================================
# MAIN
# =========================================================
def main() -> None:
    set_seed(SEED)
    df, X_ts, X_ctx = load_and_align_data()
    df = export_video_embeddings_mmaction2(df)
    df = export_kine_embeddings(df, X_ts, X_ctx)
    model = train_model(df)

    video_dim = infer_video_dim(df)
    best_model = MidLevelFusionModel(video_dim=video_dim).to(DEVICE)
    best_model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False))
    evaluate_test(best_model, df)


if __name__ == "__main__":
    main()
