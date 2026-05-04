import json
import math
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmaction.utils import register_all_modules
from mmengine.config import Config
from mmengine.runner import load_checkpoint
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# MMAction2 / MMCV imports
from mmaction.registry import MODELS
from mmcv.transforms import Compose
from mmaction.datasets.transforms import (
    DecordInit,
    DecordDecode,
    Resize,
    CenterCrop,
    FormatShape,
    PackActionInputs,
)

# =========================================================
# CONFIG
# =========================================================
KINE_X_PATH = Path("data/X_ts.npy")
KINE_META_PATH = Path("data/meta.csv")
KINE_CTX_PATH = Path("data/X_ctx.csv")
BDD_SCE_PATH = Path("data/bdd_sce.csv")
DOWNLOADED_META_PATH = Path("data/downloaded_videos_meta.csv")

VIDEO_ROOT = Path("data/annotated_videos_only")

MMACTION_CONFIG_PATH = Path("/home/nissenm27/Capstone/mmaction2/configs/recognition/slowfast/slowfast_r50_8xb8-4x16x1-256e_kinetics400-rgb.py")
MMACTION_CHECKPOINT_PATH = Path("/home/nissenm27/Capstone/mmaction2/checkpoints/slowfast_r50_8xb8-4x16x1-256e_kinetics400-rgb_20220901-701b0f6f.pth")

OUT_DIR = Path("data/mmaction2_slowfast_joint_end2end_fusion_nosplits")
OUT_DIR.mkdir(parents=True, exist_ok=True)
SPLITS_PATH = OUT_DIR / "joint_splits.csv"
MODEL_PATH = OUT_DIR / "joint_end2end_fusion_best.pth"
PRED_PATH = OUT_DIR / "joint_end2end_test_predictions.csv"
METRICS_PATH = OUT_DIR / "joint_end2end_test_metrics.json"

BATCH_SIZE = 4
EPOCHS = 20
PATIENCE = 6
LEARNING_RATE = 1e-4
BACKBONE_LEARNING_RATE = 1e-5
WEIGHT_DECAY = 1e-4
SEED = 1
NUM_CLASSES = 4
NUM_WORKERS = 0
PIN_MEMORY = False
GPU_LOG_INTERVAL = 10

FREEZE_VIDEO_BACKBONE = False
UNFREEZE_STAGE = None

# SlowFast still needs a fixed-length input tensor, so "whole clip"
# here means: sample uniformly across the full segment from BDD_START to the end.
CLIP_LEN = 32
FRAME_INTERVAL = 2
NUM_CLIPS = 1
TARGET_SIZE = 256
CROP_SIZE = 224
USE_FULL_CLIP_FROM_BDD_START = True
DEFAULT_FPS = 30.0

# Kinetics-400 normalization used by SlowFast-style RGB models.
IMG_MEAN = torch.tensor([123.675, 116.28, 103.53], dtype=torch.float32).view(1, 3, 1, 1)
IMG_STD = torch.tensor([58.395, 57.12, 57.375], dtype=torch.float32).view(1, 3, 1, 1)

# Focal loss / scheduler
FOCAL_GAMMA = 2.0
WARMUP_EPOCHS = 2
MIN_LR_SCALE = 0.05

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


def format_bytes(num_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(num_bytes)
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{num_bytes} B"


def get_gpu_stats() -> Dict[str, str]:
    if DEVICE.type != "cuda" or not torch.cuda.is_available():
        return {
            "allocated": "n/a",
            "reserved": "n/a",
            "max_allocated": "n/a",
            "max_reserved": "n/a",
        }

    return {
        "allocated": format_bytes(torch.cuda.memory_allocated(DEVICE)),
        "reserved": format_bytes(torch.cuda.memory_reserved(DEVICE)),
        "max_allocated": format_bytes(torch.cuda.max_memory_allocated(DEVICE)),
        "max_reserved": format_bytes(torch.cuda.max_memory_reserved(DEVICE)),
    }


def print_gpu_stats(prefix: str) -> None:
    stats = get_gpu_stats()
    print(
        f"{prefix} | GPU mem allocated: {stats['allocated']} | reserved: {stats['reserved']} | "
        f"peak allocated: {stats['max_allocated']} | peak reserved: {stats['max_reserved']}"
    )


# =========================================================
# LOSS
# =========================================================
class FocalLoss(nn.Module):
    def __init__(self, alpha: Optional[torch.Tensor] = None, gamma: float = 2.0) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, targets, reduction="none", weight=self.alpha)
        pt = torch.exp(-ce)
        focal = ((1.0 - pt) ** self.gamma) * ce
        return focal.mean()


# =========================================================
# KINEMATICS BRANCH
# =========================================================
class KinematicsBranch(nn.Module):
    """Trainable kinematics/context branch."""

    def __init__(self, in_chans_ts: int, in_chans_ctx: int) -> None:
        super().__init__()
        self.bn_input = nn.BatchNorm1d(in_chans_ts)
        self.conv1 = nn.Conv1d(
            in_channels=in_chans_ts,
            out_channels=64,
            kernel_size=12,
            stride=1,
            padding="same",
        )
        self.bn_conv = nn.BatchNorm1d(64)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=100,
            num_layers=2,
            batch_first=True,
            dropout=0.3,
        )

        self.ctx_fc1 = nn.Linear(in_chans_ctx, 64)
        self.ctx_relu = nn.ReLU()
        self.ctx_bn = nn.BatchNorm1d(64)
        self.ctx_drop = nn.Dropout(0.3)

    def forward(self, x_ts: torch.Tensor, x_ctx: torch.Tensor) -> torch.Tensor:
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


# =========================================================
# VIDEO BRANCH
# =========================================================
class MMAction2SlowFastFeatureExtractor(nn.Module):
    """Expose pooled SlowFast backbone embeddings from an MMAction2 recognizer."""

    def __init__(self, config_path: Path, checkpoint_path: Path, freeze_backbone: bool = False) -> None:
        super().__init__()
        if not config_path.exists():
            raise FileNotFoundError(f"MMAction2 config not found: {config_path}")
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"MMAction2 checkpoint not found: {checkpoint_path}")

        cfg = Config.fromfile(str(config_path))
        model_cfg = cfg.model

        if "backbone" in model_cfg and isinstance(model_cfg.backbone, dict):
            model_cfg.backbone.pop("pretrained", None)
            model_cfg.backbone.pop("init_cfg", None)
        model_cfg.pop("pretrained", None)
        model_cfg.pop("init_cfg", None)

        self.recognizer = MODELS.build(model_cfg)
        load_checkpoint(self.recognizer, str(checkpoint_path), map_location="cpu")
        self.backbone = self.recognizer.backbone

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        self.out_dim = self._infer_output_dim()

    @staticmethod
    def _global_pool_3d(x: torch.Tensor) -> torch.Tensor:
        return x.mean(dim=(-1, -2, -3))

    def _extract_backbone_features(self, video_inputs: torch.Tensor):
        if video_inputs.ndim != 6:
            raise ValueError(f"Expected [B, N, C, T, H, W], got {tuple(video_inputs.shape)}")
        b, n, c, t, h, w = video_inputs.shape
        flat_inputs = video_inputs.view(b * n, c, t, h, w)
        feats = self.backbone(flat_inputs)
        return feats, b, n

    def _normalize_feature_output(self, feats, b: int, n: int) -> torch.Tensor:
        if isinstance(feats, (list, tuple)) and len(feats) == 2 and all(torch.is_tensor(x) for x in feats):
            slow_feat, fast_feat = feats
            slow_vec = self._global_pool_3d(slow_feat)
            fast_vec = self._global_pool_3d(fast_feat)
            emb = torch.cat([slow_vec, fast_vec], dim=1)
        elif torch.is_tensor(feats):
            emb = self._global_pool_3d(feats)
        elif isinstance(feats, dict):
            tensor_values = [v for v in feats.values() if torch.is_tensor(v)]
            if not tensor_values:
                raise RuntimeError("Backbone returned dict with no tensor values.")
            pooled = [self._global_pool_3d(v) for v in tensor_values]
            emb = torch.cat(pooled, dim=1)
        else:
            raise RuntimeError(f"Unsupported SlowFast feature type: {type(feats)}")

        return emb.view(b, n, -1).mean(dim=1)

    def _infer_output_dim(self) -> int:
        self.eval()
        with torch.no_grad():
            dummy = torch.randn(1, 1, 3, CLIP_LEN, CROP_SIZE, CROP_SIZE)
            feats, b, n = self._extract_backbone_features(dummy)
            emb = self._normalize_feature_output(feats, b, n)
        return int(emb.shape[1])

    def forward(self, video_inputs: torch.Tensor) -> torch.Tensor:
        feats, b, n = self._extract_backbone_features(video_inputs)
        return self._normalize_feature_output(feats, b, n)


# =========================================================
# JOINT MODEL
# =========================================================
class JointEndToEndFusionModel(nn.Module):
    def __init__(
        self,
        video_encoder: MMAction2SlowFastFeatureExtractor,
        in_chans_ts: int,
        in_chans_ctx: int,
        num_classes: int = NUM_CLASSES,
    ) -> None:
        super().__init__()
        self.video_encoder = video_encoder
        self.kine_encoder = KinematicsBranch(in_chans_ts=in_chans_ts, in_chans_ctx=in_chans_ctx)

        video_dim = video_encoder.out_dim
        kine_dim = 164

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

    def forward(self, video_inputs: torch.Tensor, x_ts: torch.Tensor, x_ctx: torch.Tensor) -> torch.Tensor:
        video_embedding = self.video_encoder(video_inputs)
        kine_embedding = self.kine_encoder(x_ts, x_ctx)

        v = self.video_proj(video_embedding)
        k = self.kine_proj(kine_embedding)
        joint = torch.cat([v, k, v * k, torch.abs(v - k)], dim=1)
        fused = self.fusion(joint)
        return self.classifier(fused)


# =========================================================
# DATA PREP
# =========================================================
def build_video_path(bdd_id: str) -> Optional[str]:
    candidates = [
        VIDEO_ROOT / f"{bdd_id}.mov",
        VIDEO_ROOT / f"{bdd_id}.mp4",
        VIDEO_ROOT / f"{bdd_id}.MOV",
        VIDEO_ROOT / f"{bdd_id}.MP4",
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return None


def build_base_dataframe() -> pd.DataFrame:
    meta = pd.read_csv(KINE_META_PATH).copy()
    if "BDD_ID" not in meta.columns or "y" not in meta.columns:
        raise ValueError("meta.csv must contain BDD_ID and y.")

    meta["BDD_ID"] = meta["BDD_ID"].astype(str)
    meta["row_idx"] = np.arange(len(meta))
    meta["target_idx"] = meta["y"].astype(int)
    meta["video_path"] = meta["BDD_ID"].map(build_video_path)

    if BDD_SCE_PATH.exists():
        bdd_sce = pd.read_csv(BDD_SCE_PATH).copy()
        if "BDD_ID" in bdd_sce.columns and "BDD_START" in bdd_sce.columns:
            bdd_sce["BDD_ID"] = bdd_sce["BDD_ID"].astype(str)
            bdd_sce = (
                bdd_sce[["BDD_ID", "BDD_START"]]
                .dropna(subset=["BDD_ID"])
                .drop_duplicates(subset=["BDD_ID"], keep="first")
            )
            meta = meta.merge(bdd_sce, on="BDD_ID", how="left")
        else:
            meta["BDD_START"] = 0.0
    else:
        meta["BDD_START"] = 0.0

    meta["BDD_START"] = meta["BDD_START"].fillna(0.0).astype(float).clip(lower=0.0)

    df = meta.dropna(subset=["video_path"]).drop_duplicates(subset=["BDD_ID"]).reset_index(drop=True)
    if len(df) == 0:
        raise ValueError(f"No matching videos found in {VIDEO_ROOT} for BDD_ID values in meta.csv")
    return df


def build_group_splits(df: pd.DataFrame) -> pd.DataFrame:
    groups = df["BDD_ID"].astype(str).to_numpy()
    y = df["target_idx"].to_numpy()

    gss1 = GroupShuffleSplit(n_splits=1, train_size=0.70, random_state=SEED)
    train_idx, temp_idx = next(gss1.split(df, y, groups=groups))

    temp_df = df.iloc[temp_idx].copy()
    temp_groups = temp_df["BDD_ID"].astype(str).to_numpy()
    temp_y = temp_df["target_idx"].to_numpy()

    gss2 = GroupShuffleSplit(n_splits=1, train_size=0.50, random_state=SEED)
    val_rel, test_rel = next(gss2.split(temp_df, temp_y, groups=temp_groups))

    df = df.copy()
    df["split"] = "train"
    df.iloc[temp_idx, df.columns.get_loc("split")] = "temp"
    df.iloc[temp_idx[val_rel], df.columns.get_loc("split")] = "val"
    df.iloc[temp_idx[test_rel], df.columns.get_loc("split")] = "test"
    return df.reset_index(drop=True)


def normalize_time_series_train_only(X_ts_sel: np.ndarray, train_mask: np.ndarray) -> np.ndarray:
    """Per-channel standardization over all train timesteps only."""
    if X_ts_sel.ndim != 3:
        raise ValueError(f"Expected X_ts shape [N, C, T], got {X_ts_sel.shape}")

    n, c, t = X_ts_sel.shape
    x_train = X_ts_sel[train_mask].transpose(0, 2, 1).reshape(-1, c)
    scaler = StandardScaler()
    scaler.fit(x_train)

    x_all = X_ts_sel.transpose(0, 2, 1).reshape(-1, c)
    x_all = scaler.transform(x_all)
    x_all = x_all.reshape(n, t, c).transpose(0, 2, 1).astype(np.float32)
    return x_all


def load_and_align_data() -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    df = build_base_dataframe()
    df = build_group_splits(df)

    meta = pd.read_csv(KINE_META_PATH)
    ctx = pd.read_csv(KINE_CTX_PATH)
    X_ts = np.load(KINE_X_PATH).astype(np.float32)

    assert len(meta) == len(ctx) == len(X_ts), "Mismatch among meta, ctx, and X_ts lengths"

    ctx_features = ctx.drop(columns=[c for c in ["BDD_ID", "EVENT_ID", "EVENT_TYPE", "y"] if c in ctx.columns])
    categorical_cols = [c for c in ["weather", "scene", "timeofday"] if c in ctx_features.columns]
    ctx_features = pd.get_dummies(ctx_features, columns=categorical_cols, dummy_na=True)
    ctx_features = ctx_features.fillna(0)
    X_ctx_all = ctx_features.to_numpy(dtype=np.float32)

    row_idx = df["row_idx"].to_numpy(dtype=int)
    X_ts_sel = X_ts[row_idx]
    X_ctx_sel = X_ctx_all[row_idx]

    train_mask = df["split"].to_numpy() == "train"

    # Normalize X_ts using train split statistics only.
    X_ts_sel = normalize_time_series_train_only(X_ts_sel, train_mask)

    # Normalize context using train split statistics only.
    ctx_scaler = StandardScaler()
    X_ctx_sel[train_mask] = ctx_scaler.fit_transform(X_ctx_sel[train_mask])
    X_ctx_sel[~train_mask] = ctx_scaler.transform(X_ctx_sel[~train_mask])

    df.to_csv(SPLITS_PATH, index=False)
    print(f"Saved fresh split metadata to: {SPLITS_PATH}")
    return df, X_ts_sel, X_ctx_sel


# =========================================================
# VIDEO PREPROCESS
# =========================================================
class SimpleSlowFastVideoLoader:
    """Decode a video into [N, C, T, H, W] and sample from BDD_START onward."""

    def __init__(self) -> None:
        self.init_pipeline = Compose([DecordInit()])
        self.post_decode = Compose([
            Resize(scale=(-1, TARGET_SIZE)),
            CenterCrop(crop_size=CROP_SIZE),
            FormatShape(input_format="NCTHW"),
            PackActionInputs(),
        ])

    @staticmethod
    def _safe_get_fps(video_reader) -> float:
        try:
            fps = float(video_reader.get_avg_fps())
            if fps > 1e-6:
                return fps
        except Exception:
            pass
        return DEFAULT_FPS

    def _sample_frame_indices(self, total_frames: int, fps: float, bdd_start_sec: float) -> np.ndarray:
        if total_frames <= 0:
            raise RuntimeError("Video has no frames.")

        start_frame = int(round(max(bdd_start_sec, 0.0) * fps))
        start_frame = min(max(start_frame, 0), max(total_frames - 1, 0))

        if USE_FULL_CLIP_FROM_BDD_START:
            end_frame = max(total_frames - 1, start_frame)
            if end_frame == start_frame:
                return np.full((CLIP_LEN,), start_frame, dtype=int)
            idx = np.linspace(start_frame, end_frame, num=CLIP_LEN)
            return np.clip(np.round(idx).astype(int), 0, total_frames - 1)

        span = CLIP_LEN * FRAME_INTERVAL
        if total_frames - start_frame <= span:
            idx = np.linspace(start_frame, total_frames - 1, num=CLIP_LEN)
        else:
            idx = start_frame + np.arange(CLIP_LEN) * FRAME_INTERVAL
        return np.clip(np.round(idx).astype(int), 0, total_frames - 1)

    @staticmethod
    def _normalize_video_tensor(video_tensor: torch.Tensor) -> torch.Tensor:
        # [N, C, T, H, W], float32 in [0,255] after decode/pack.
        video_tensor = video_tensor.float()
        mean = IMG_MEAN.to(device=video_tensor.device, dtype=video_tensor.dtype)
        std = IMG_STD.to(device=video_tensor.device, dtype=video_tensor.dtype)
        return (video_tensor - mean) / std

    def __call__(self, video_path: str, bdd_start_sec: float) -> torch.Tensor:
        results = {"filename": video_path, "start_index": 0, "modality": "RGB"}
        results = self.init_pipeline(results)
        total_frames = int(results["total_frames"])
        fps = self._safe_get_fps(results["video_reader"])
        frame_inds = self._sample_frame_indices(total_frames=total_frames, fps=fps, bdd_start_sec=bdd_start_sec)

        decode_results = {
            "filename": video_path,
            "video_reader": results["video_reader"],
            "total_frames": total_frames,
            "frame_inds": frame_inds,
            "clip_len": CLIP_LEN,
            "frame_interval": FRAME_INTERVAL,
            "num_clips": NUM_CLIPS,
            "start_index": 0,
            "modality": "RGB",
        }
        decode_results = DecordDecode()(decode_results)
        packed = self.post_decode(decode_results)
        video_inputs = packed["inputs"].float()
        return self._normalize_video_tensor(video_inputs)


# =========================================================
# DATASETS
# =========================================================
class JointFusionDataset(Dataset):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        X_ts: np.ndarray,
        X_ctx: np.ndarray,
        video_loader: SimpleSlowFastVideoLoader,
    ) -> None:
        self.df = dataframe.reset_index(drop=True)
        self.X_ts = torch.tensor(X_ts, dtype=torch.float32)
        self.X_ctx = torch.tensor(X_ctx, dtype=torch.float32)
        self.video_loader = video_loader

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        video_path = str(row["video_path"])
        if not Path(video_path).exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        bdd_start = float(row.get("BDD_START", 0.0))
        video_inputs = self.video_loader(video_path, bdd_start)
        x_ts = self.X_ts[idx]
        x_ctx = self.X_ctx[idx]
        target = int(row["target_idx"])
        bdd_id = str(row["BDD_ID"])
        return video_inputs, x_ts, x_ctx, target, bdd_id


def joint_collate_fn(batch):
    videos, xs_ts, xs_ctx, targets, ids = zip(*batch)
    videos = torch.stack(videos, dim=0)
    xs_ts = torch.stack(xs_ts, dim=0)
    xs_ctx = torch.stack(xs_ctx, dim=0)
    targets = torch.tensor(targets, dtype=torch.long)
    return videos, xs_ts, xs_ctx, targets, list(ids)


# =========================================================
# TRAIN / EVAL
# =========================================================
def make_loaders(df: pd.DataFrame, X_ts: np.ndarray, X_ctx: np.ndarray):
    train_df = df[df["split"] == "train"].copy()
    val_df = df[df["split"] == "val"].copy()
    test_df = df[df["split"] == "test"].copy()

    video_loader = SimpleSlowFastVideoLoader()

    train_loader = DataLoader(
        JointFusionDataset(train_df, X_ts[df["split"] == "train"], X_ctx[df["split"] == "train"], video_loader),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        collate_fn=joint_collate_fn,
        drop_last=True,
    )
    val_loader = DataLoader(
        JointFusionDataset(val_df, X_ts[df["split"] == "val"], X_ctx[df["split"] == "val"], video_loader),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        collate_fn=joint_collate_fn,
    )
    test_loader = DataLoader(
        JointFusionDataset(test_df, X_ts[df["split"] == "test"], X_ctx[df["split"] == "test"], video_loader),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        collate_fn=joint_collate_fn,
    )
    return train_loader, val_loader, test_loader


def maybe_freeze_partial(model: nn.Module) -> None:
    if FREEZE_VIDEO_BACKBONE:
        for p in model.video_encoder.backbone.parameters():
            p.requires_grad = False
        return

    if UNFREEZE_STAGE is not None:
        for name, p in model.video_encoder.backbone.named_parameters():
            p.requires_grad = UNFREEZE_STAGE in name


def build_optimizer_and_scheduler(model: nn.Module, total_steps: int):
    backbone_params = []
    head_params = []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith("video_encoder.backbone"):
            backbone_params.append(p)
        else:
            head_params.append(p)

    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": BACKBONE_LEARNING_RATE, "weight_decay": WEIGHT_DECAY},
            {"params": head_params, "lr": LEARNING_RATE, "weight_decay": WEIGHT_DECAY},
        ]
    )

    warmup_steps = max(WARMUP_EPOCHS * total_steps, 1)
    all_steps = max(EPOCHS * total_steps, 1)

    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return float(current_step + 1) / float(warmup_steps)

        progress = float(current_step - warmup_steps) / float(max(all_steps - warmup_steps, 1))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return MIN_LR_SCALE + (1.0 - MIN_LR_SCALE) * cosine

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    return optimizer, scheduler


def evaluate_loader(model: nn.Module, loader: DataLoader) -> Tuple[float, List[int], List[int], List[str]]:
    model.eval()
    y_true: List[int] = []
    y_pred: List[int] = []
    ids: List[str] = []
    eval_start = time.perf_counter()

    if DEVICE.type == "cuda":
        torch.cuda.reset_peak_memory_stats(DEVICE)

    with torch.no_grad():
        for batch_idx, (batch_v, batch_ts, batch_ctx, batch_targets, batch_ids) in enumerate(tqdm(loader, desc="Evaluating", leave=False), start=1):
            batch_start = time.perf_counter()
            batch_v = batch_v.to(DEVICE)
            batch_ts = batch_ts.to(DEVICE)
            batch_ctx = batch_ctx.to(DEVICE)
            logits = model(batch_v, batch_ts, batch_ctx)
            if DEVICE.type == "cuda":
                torch.cuda.synchronize(DEVICE)
            batch_time = time.perf_counter() - batch_start
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            y_true.extend(batch_targets.numpy().tolist())
            y_pred.extend(preds.tolist())
            ids.extend(list(batch_ids))

            if batch_idx % GPU_LOG_INTERVAL == 0 or batch_idx == len(loader):
                stats = get_gpu_stats()
                print(
                    f"Eval batch {batch_idx:03d}/{len(loader):03d} | batch time: {batch_time:.2f}s | "
                    f"GPU alloc: {stats['allocated']} | GPU reserved: {stats['reserved']} | "
                    f"peak alloc: {stats['max_allocated']}"
                )

    total_time = time.perf_counter() - eval_start
    print_gpu_stats(f"Evaluation complete in {total_time:.2f}s")
    return accuracy_score(y_true, y_pred), y_true, y_pred, ids


def train_model(df: pd.DataFrame, X_ts: np.ndarray, X_ctx: np.ndarray) -> JointEndToEndFusionModel:
    train_loader, val_loader, _ = make_loaders(df, X_ts, X_ctx)

    video_encoder = MMAction2SlowFastFeatureExtractor(
        config_path=MMACTION_CONFIG_PATH,
        checkpoint_path=MMACTION_CHECKPOINT_PATH,
        freeze_backbone=FREEZE_VIDEO_BACKBONE,
    )
    model = JointEndToEndFusionModel(
        video_encoder=video_encoder,
        in_chans_ts=X_ts.shape[1],
        in_chans_ctx=X_ctx.shape[1],
        num_classes=NUM_CLASSES,
    ).to(DEVICE)
    maybe_freeze_partial(model)

    train_targets = df[df["split"] == "train"]["target_idx"].to_numpy(dtype=int)
    class_counts = np.bincount(train_targets, minlength=NUM_CLASSES)
    class_weights = len(train_targets) / (NUM_CLASSES * np.maximum(class_counts, 1))
    class_weights = torch.tensor(class_weights, dtype=torch.float32, device=DEVICE)

    criterion = FocalLoss(alpha=class_weights, gamma=FOCAL_GAMMA)
    optimizer, scheduler = build_optimizer_and_scheduler(model, total_steps=len(train_loader))

    best_val_acc = -1.0
    patience_ctr = 0
    history: List[Dict[str, float]] = []
    global_step = 0

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        epoch_start = time.perf_counter()

        if DEVICE.type == "cuda":
            torch.cuda.reset_peak_memory_stats(DEVICE)

        for batch_idx, (batch_v, batch_ts, batch_ctx, batch_targets, _) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1:02d}/{EPOCHS}", leave=False), start=1):
            batch_start = time.perf_counter()
            batch_v = batch_v.to(DEVICE)
            batch_ts = batch_ts.to(DEVICE)
            batch_ctx = batch_ctx.to(DEVICE)
            batch_targets = batch_targets.to(DEVICE)

            optimizer.zero_grad()
            logits = model(batch_v, batch_ts, batch_ctx)
            loss = criterion(logits, batch_targets)
            loss.backward()
            optimizer.step()
            scheduler.step()
            global_step += 1

            if DEVICE.type == "cuda":
                torch.cuda.synchronize(DEVICE)
            batch_time = time.perf_counter() - batch_start

            running_loss += loss.item() * batch_targets.size(0)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == batch_targets).sum().item()
            total += batch_targets.size(0)

            if batch_idx % GPU_LOG_INTERVAL == 0 or batch_idx == len(train_loader):
                stats = get_gpu_stats()
                backbone_lr = optimizer.param_groups[0]["lr"]
                head_lr = optimizer.param_groups[1]["lr"]
                print(
                    f"Epoch {epoch + 1:02d} batch {batch_idx:03d}/{len(train_loader):03d} | "
                    f"batch time: {batch_time:.2f}s | samples: {batch_targets.size(0)} | "
                    f"backbone_lr: {backbone_lr:.2e} | head_lr: {head_lr:.2e} | "
                    f"GPU alloc: {stats['allocated']} | GPU reserved: {stats['reserved']} | "
                    f"peak alloc: {stats['max_allocated']}"
                )

        train_loss = running_loss / max(total, 1)
        train_acc = correct / max(total, 1)
        epoch_time = time.perf_counter() - epoch_start
        samples_per_sec = total / max(epoch_time, 1e-8)
        val_acc, _, _, _ = evaluate_loader(model, val_loader)
        history.append({"epoch": epoch + 1, "train_loss": train_loss, "train_acc": train_acc, "val_acc": val_acc})
        print(
            f"Epoch {epoch + 1:02d} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Val Acc: {val_acc:.4f} | Epoch Time: {epoch_time:.2f}s | Throughput: {samples_per_sec:.2f} samples/s"
        )
        print_gpu_stats(f"Epoch {epoch + 1:02d} GPU summary")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_ctr = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "history": history,
                    "video_dim": model.video_encoder.out_dim,
                },
                MODEL_PATH,
            )
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                print(f"Early stopping at epoch {epoch + 1}.")
                break

    print(f"\nBest joint end-to-end fusion model saved to: {MODEL_PATH}")
    return model


def evaluate_test(df: pd.DataFrame, X_ts: np.ndarray, X_ctx: np.ndarray) -> None:
    _, _, test_loader = make_loaders(df, X_ts, X_ctx)

    video_encoder = MMAction2SlowFastFeatureExtractor(
        config_path=MMACTION_CONFIG_PATH,
        checkpoint_path=MMACTION_CHECKPOINT_PATH,
        freeze_backbone=False,
    )
    model = JointEndToEndFusionModel(
        video_encoder=video_encoder,
        in_chans_ts=X_ts.shape[1],
        in_chans_ctx=X_ctx.shape[1],
        num_classes=NUM_CLASSES,
    ).to(DEVICE)

    ckpt = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    acc, y_true, y_pred, ids = evaluate_loader(model, test_loader)

    print("\n" + "=" * 80)
    print("JOINT END-TO-END MMACTION2 SLOWFAST FUSION TEST RESULTS")
    print(f"Test Accuracy: {acc:.4f}")
    report = classification_report(
        y_true,
        y_pred,
        target_names=[CLASS_MAP_STR[i] for i in range(NUM_CLASSES)],
        output_dict=True,
    )
    print(classification_report(y_true, y_pred, target_names=[CLASS_MAP_STR[i] for i in range(NUM_CLASSES)]))

    out = pd.DataFrame(
        {
            "BDD_ID": ids,
            "true_idx": y_true,
            "pred_idx": y_pred,
            "true_label": [CLASS_MAP_STR[i] for i in y_true],
            "pred_label": [CLASS_MAP_STR[i] for i in y_pred],
        }
    )
    out.to_csv(PRED_PATH, index=False)

    with open(METRICS_PATH, "w") as f:
        json.dump({"test_accuracy": acc, "classification_report": report}, f, indent=2)

    print(f"Saved predictions to: {PRED_PATH}")
    print(f"Saved metrics to: {METRICS_PATH}")
    print("=" * 80)


# =========================================================
# MAIN
# =========================================================
def main() -> None:
    register_all_modules()
    set_seed(SEED)
    df, X_ts, X_ctx = load_and_align_data()
    train_model(df, X_ts, X_ctx)
    evaluate_test(df, X_ts, X_ctx)


if __name__ == "__main__":
    main()
