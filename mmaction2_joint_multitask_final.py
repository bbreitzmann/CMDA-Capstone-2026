# ============================================================
# Four-head MMAction2 SlowFast + kinematics pipeline
# ============================================================
# This script trains/evaluates four heads: event type, grouped conflict,
# exact 17-way conflict type, and BDD_START regression.

# imports
import json
import math
import os
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
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    mean_absolute_error,
    mean_squared_error,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


# These imports build the video decoding/preprocessing pipeline and the
# pretrained MMAction2 SlowFast recognizer backbone.
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


# ============================================================
# CONFIG
# ============================================================
# Input data paths for kinematics, metadata, context features, and SCE annotations
KINE_X_PATH = Path("data/X_ts.npy")
KINE_META_PATH = Path("data/meta.csv")
KINE_CTX_PATH = Path("data/X_ctx.csv")
BDD_SCE_PATH = Path("data/bdd_sce.csv")

# Folder containing the actual .mov/.mp4 videos used by the video branch
VIDEO_ROOT = Path("data/annotated_videos_only")

# MMAction2 SlowFast model config and Kinetics-400 pretrained checkpoint
MMACTION_CONFIG_PATH = Path(
    "/home/nissenm27/Capstone/mmaction2/configs/recognition/slowfast/slowfast_r50_8xb8-4x16x1-256e_kinetics400-rgb.py"
)
MMACTION_CHECKPOINT_PATH = Path(
    "/home/nissenm27/Capstone/mmaction2/checkpoints/slowfast_r50_8xb8-4x16x1-256e_kinetics400-rgb_20220901-701b0f6f.pth"
)

# Experiment output folder. Model checkpoints, predictions, metrics, and splits are saved here
OUT_DIR = Path("data/mmaction2_slowfast_joint_multitask_four_head")
OUT_DIR.mkdir(parents=True, exist_ok=True)
SPLITS_PATH = OUT_DIR / "joint_splits.csv"
MODEL_PATH = OUT_DIR / "joint_multitask_four_head_best.pth"
PRED_PATH = OUT_DIR / "joint_multitask_four_head_test_predictions.csv"
METRICS_PATH = OUT_DIR / "joint_multitask_four_head_test_metrics.json"

# Training hyperparameters. Batch size is small because video tensors are memory-heavy.
BATCH_SIZE = 4
EPOCHS = 40
PATIENCE = 8
LEARNING_RATE = 1e-4
BACKBONE_LEARNING_RATE = 1e-5
WEIGHT_DECAY = 1e-4
SEED = 1
AUX_WARMUP_EPOCHS = 5

# Main event task has 4 classes: Conflict, Bump, Hard Brake, Not an SCE.
NUM_EVENT_CLASSES = 4
# Broad conflict group mapping. These five groups are an auxiliary helper task.
CONFLICT_GROUPS = {
    "vehicle_proximity": ["Q", "R", "T", "U"],
    "vehicle_path_conflict": ["E", "Y", "I", "O", "A", "S"],
    "vru": ["D", "F"],
    "road_hazard": ["W", "P", "G", "H"],
    "other": ["J"],
}
GROUP_NAMES = list(CONFLICT_GROUPS.keys())
CONFLICT_TO_GROUP = {
    code: group_name
    for group_name, codes in CONFLICT_GROUPS.items()
    for code in codes
}
GROUP_TO_IDX = {group_name: idx for idx, group_name in enumerate(GROUP_NAMES)}
IDX_TO_GROUP = {idx: group_name for group_name, idx in GROUP_TO_IDX.items()}
NUM_CONFLICT_GROUP_CLASSES = len(GROUP_NAMES)
IGNORE_CONFLICT_GROUP_INDEX = -100

# Exact 17-way conflict labels. This is separate from the grouped conflict head.
CONFLICT_TYPES_17 = ["Q", "W", "E", "R", "T", "Y", "U", "I", "O", "P", "A", "S", "D", "F", "G", "H", "J"]
CONFLICT17_TO_IDX = {code: idx for idx, code in enumerate(CONFLICT_TYPES_17)}
IDX_TO_CONFLICT17 = {idx: code for code, idx in CONFLICT17_TO_IDX.items()}
NUM_CONFLICT_17_CLASSES = len(CONFLICT_TYPES_17)
IGNORE_CONFLICT_17_INDEX = -100

# Runtime flags for DataLoader memory pinning, GPU logging, and mixed precision
PIN_MEMORY = torch.cuda.is_available()
GPU_LOG_INTERVAL = 10
USE_AMP = torch.cuda.is_available()

# Fine-tuning controls for the pretrained SlowFast video backbone
FREEZE_VIDEO_BACKBONE = False
UNFREEZE_STAGE = None
# If True, the exact 17-way conflict loss does not backprop into shared fusion features
DETACH_CONFLICT17_HEAD = True

# Video sampling settings The loader samples a fixed-length clip from BDD_START onward
CLIP_LEN = 64
FRAME_INTERVAL = 2
NUM_CLIPS = 1
TARGET_SIZE = 256
CROP_SIZE = 224
USE_FULL_CLIP_FROM_BDD_START = True
DEFAULT_FPS = 30.0

# RGB normalization expected by the pretrained SlowFast model
IMG_MEAN = torch.tensor([123.675, 116.28, 103.53], dtype=torch.float32).view(1, 3, 1, 1, 1)
IMG_STD = torch.tensor([58.395, 57.12, 57.375], dtype=torch.float32).view(1, 3, 1, 1, 1)

# Focal loss and learning-rate scheduler settings
FOCAL_GAMMA = 2.0
WARMUP_EPOCHS = 2
MIN_LR_SCALE = 0.05

# Multi-task loss weights. Event is primary, auxiliary heads are weighted smaller
LAMBDA_EVENT = 1.0
LAMBDA_CONFLICT_GROUP = 0.15
LAMBDA_CONFLICT_17 = 0.03
LAMBDA_START = 0.05

# Human-readable names for event class indices.
CLASS_MAP_STR = {
    0: "Conflict",
    1: "Bump",
    2: "Hard Brake",
    3: "Not an SCE",
}

# Select GPU when available; otherwise run on CPU.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {DEVICE.type.upper()} device!")
if DEVICE.type == "cuda":
    torch.backends.cudnn.benchmark = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass


# ============================================================
# UTILS
# ============================================================
# Set NumPy and PyTorch seeds to make runs more reproducible
def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# Count CPUs visible to the job
def get_visible_cpu_count() -> int:
    try:
        return len(os.sched_getaffinity(0))
    except Exception:
        return os.cpu_count() or 1


# Pick DataLoader worker count based on allocated CPUs, capped at 12
NUM_WORKERS = min(12, max(get_visible_cpu_count() - 2, 1))


# Convert memory counts from bytes into readable B/KB/MB/GB/TB strings
def format_bytes(num_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(num_bytes)
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{num_bytes} B"


# Read current and peak CUDA memory stats for logging
def get_gpu_stats() -> Dict[str, str]:
    if DEVICE.type != "cuda" or not torch.cuda.is_available():
        return {"allocated": "n/a", "reserved": "n/a", "max_allocated": "n/a", "max_reserved": "n/a"}
    return {
        "allocated": format_bytes(torch.cuda.memory_allocated(DEVICE)),
        "reserved": format_bytes(torch.cuda.memory_reserved(DEVICE)),
        "max_allocated": format_bytes(torch.cuda.max_memory_allocated(DEVICE)),
        "max_reserved": format_bytes(torch.cuda.max_memory_reserved(DEVICE)),
    }

# Print GPU stats with a prefix describing the current stage
def print_gpu_stats(prefix: str) -> None:
    stats = get_gpu_stats()
    print(
        f"{prefix} | GPU mem allocated: {stats['allocated']} | reserved: {stats['reserved']} | "
        f"peak allocated: {stats['max_allocated']} | peak reserved: {stats['max_reserved']}"
    )


# Normalize raw conflict labels and reject anything not in the 17 valid conflict types
def safe_upper_conflict_code(value) -> Optional[str]:
    if pd.isna(value):
        return None
    code = str(value).strip().upper()
    if code in CONFLICT17_TO_IDX:
        return code
    return None


# ============================================================
# LOSSES
# ============================================================
# Focal loss emphasizes hard examples and helps with imbalanced event classes
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


# Compute the total weighted loss across event, grouped conflict, exact conflict, and start time
# Conflict losses are masked so they only train on true Conflict rows with valid labels
def compute_multitask_loss(
    outputs: Dict[str, torch.Tensor],
    event_targets: torch.Tensor,
    conflict_group_targets: torch.Tensor,
    conflict17_targets: torch.Tensor,
    start_targets: torch.Tensor,
    event_criterion: nn.Module,
    conflict_group_criterion: nn.Module,
    conflict17_criterion: nn.Module,
    start_criterion: nn.Module,
    epoch: int,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    event_loss = event_criterion(outputs["event_logits"], event_targets)
    start_loss = start_criterion(outputs["start_pred"], start_targets)

    group_mask = (event_targets == 0) & (conflict_group_targets >= 0)
    if group_mask.any():
        conflict_group_loss = conflict_group_criterion(
            outputs["conflict_group_logits"][group_mask],
            conflict_group_targets[group_mask],
        )
    else:
        conflict_group_loss = torch.zeros((), device=event_loss.device, dtype=event_loss.dtype)

    c17_mask = (event_targets == 0) & (conflict17_targets >= 0)
    if c17_mask.any():
        conflict17_loss = conflict17_criterion(
            outputs["conflict17_logits"][c17_mask],
            conflict17_targets[c17_mask],
        )
    else:
        conflict17_loss = torch.zeros((), device=event_loss.device, dtype=event_loss.dtype)

    current_group_weight = LAMBDA_CONFLICT_GROUP if epoch >= AUX_WARMUP_EPOCHS else 0.0
    current_c17_weight = LAMBDA_CONFLICT_17 if epoch >= AUX_WARMUP_EPOCHS else 0.0

    total = (
        LAMBDA_EVENT * event_loss
        + current_group_weight * conflict_group_loss
        + current_c17_weight * conflict17_loss
        + LAMBDA_START * start_loss
    )
    info = {
        "total": float(total.item()),
        "event": float(event_loss.item()),
        "conflict_group": float(conflict_group_loss.item()),
        "conflict17": float(conflict17_loss.item()),
        "start": float(start_loss.item()),
        "n_conflict_group": int(group_mask.sum().item()),
        "n_conflict17": int(c17_mask.sum().item()),
        "current_group_weight": float(current_group_weight),
        "current_c17_weight": float(current_c17_weight),
    }
    return total, info

# ============================================================
# KINEMATICS BRANCH
# ============================================================
# Encodes synchronized vehicle motion time-series and context variables
# Learnable temporal pooling: assigns attention weight to each timestep
class TemporalAttentionPool(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.score = nn.Linear(dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = torch.softmax(self.score(x).squeeze(-1), dim=1)
        return torch.sum(x * w.unsqueeze(-1), dim=1)


# Richer kinematics branch using Conv1D, BiLSTM, attention pooling, summary features, and context features.
class KinematicsBranchV2(nn.Module):
    def __init__(self, in_chans_ts: int, in_chans_ctx: int) -> None:
        super().__init__()
        self.ts_norm = nn.BatchNorm1d(in_chans_ts)
        self.conv_stack = nn.Sequential(
            nn.Conv1d(in_chans_ts, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.15),
        )
        self.bilstm = nn.LSTM(
            input_size=128,
            hidden_size=96,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.attn_pool = TemporalAttentionPool(dim=192)
        self.summary_proj = nn.Sequential(
            nn.Linear(in_chans_ts * 8, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 96),
            nn.ReLU(),
        )
        self.ctx_proj = nn.Sequential(
            nn.Linear(in_chans_ctx, 96),
            nn.BatchNorm1d(96),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        self.out_dim = 192 + 96 + 96

    @staticmethod
    def build_summary_features(x_ts: torch.Tensor) -> torch.Tensor:
        start = x_ts[:, :, 0]
        end = x_ts[:, :, -1]
        mean = x_ts.mean(dim=2)
        std = x_ts.std(dim=2, unbiased=False)
        minv = x_ts.min(dim=2).values
        maxv = x_ts.max(dim=2).values
        delta = end - start
        rangev = maxv - minv
        if x_ts.shape[2] > 1:
            diffs = x_ts[:, :, 1:] - x_ts[:, :, :-1]
            abs_diff_mean = diffs.abs().mean(dim=2)
        else:
            abs_diff_mean = torch.zeros_like(mean)
        return torch.cat([mean, std, minv, maxv, delta, rangev, start, abs_diff_mean], dim=1)

    def forward(self, x_ts: torch.Tensor, x_ctx: torch.Tensor) -> torch.Tensor:
        x_ts = self.ts_norm(x_ts)
        conv = self.conv_stack(x_ts)
        seq = conv.permute(0, 2, 1)
        lstm_out, _ = self.bilstm(seq)
        ts_repr = self.attn_pool(lstm_out)
        summary_repr = self.summary_proj(self.build_summary_features(x_ts))
        ctx_repr = self.ctx_proj(x_ctx)
        return torch.cat([ts_repr, summary_repr, ctx_repr], dim=1)


# ============================================================
# VIDEO BRANCH
# ============================================================
# Wraps MMAction2 SlowFast so the pipeline can use pretrained video features
# Builds the MMAction2 recognizer, loads the checkpoint, and exposes pooled backbone embeddings
class MMAction2SlowFastFeatureExtractor(nn.Module):
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
            emb = torch.cat([self._global_pool_3d(slow_feat), self._global_pool_3d(fast_feat)], dim=1)
        elif torch.is_tensor(feats):
            emb = self._global_pool_3d(feats)
        elif isinstance(feats, dict):
            vals = [v for v in feats.values() if torch.is_tensor(v)]
            if not vals:
                raise RuntimeError("Backbone returned dict with no tensor values.")
            emb = torch.cat([self._global_pool_3d(v) for v in vals], dim=1)
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


# ============================================================
# JOINT MULTI-TASK MODEL
# ============================================================
# Multimodal fusion model with four task-specific heads
# Full model is video encoder + kinematics encoder + fusion trunk + four output heads
class JointMultiTaskFourHeadModel(nn.Module):
    """Four-head version of the current updated pipeline

    Heads:
      1) event head (4-way)
      2) grouped conflict head (5-way helper task)
      3) 17-way conflict type head (autolabeler output)
      4) start-time regression head
    """

    def __init__(
        self,
        video_encoder: MMAction2SlowFastFeatureExtractor,
        in_chans_ts: int,
        in_chans_ctx: int,
    ) -> None:
        super().__init__()
        self.video_encoder = video_encoder
        self.kine_encoder = KinematicsBranchV2(in_chans_ts=in_chans_ts, in_chans_ctx=in_chans_ctx)

        video_dim = video_encoder.out_dim
        kine_dim = self.kine_encoder.out_dim

        self.video_proj = nn.Sequential(
            nn.Linear(video_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(512, 256),
            nn.ReLU(),
        )
        self.kine_proj = nn.Sequential(
            nn.Linear(kine_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.20),
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

        self.event_head = nn.Linear(128, NUM_EVENT_CLASSES)
        self.conflict_group_head = nn.Sequential(
            nn.Linear(128 + 256 + 256, 256),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(256, NUM_CONFLICT_GROUP_CLASSES),
        )
        self.conflict17_head = nn.Sequential(
            nn.Linear(128 + 256 + 256, 256),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(256, NUM_CONFLICT_17_CLASSES),
        )
        self.start_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
        )

    def forward(self, video_inputs: torch.Tensor, x_ts: torch.Tensor, x_ctx: torch.Tensor) -> Dict[str, torch.Tensor]:
        video_embedding = self.video_encoder(video_inputs)
        kine_embedding = self.kine_encoder(x_ts, x_ctx)
        v = self.video_proj(video_embedding)
        k = self.kine_proj(kine_embedding)
        joint = torch.cat([v, k, v * k, torch.abs(v - k)], dim=1)
        fused = self.fusion(joint)

        shared_task_features = torch.cat([fused, v, k], dim=1)
        if DETACH_CONFLICT17_HEAD:
            conflict17_features = torch.cat([fused.detach(), v.detach(), k.detach()], dim=1)
        else:
            conflict17_features = shared_task_features

        return {
            "event_logits": self.event_head(fused),
            "conflict_group_logits": self.conflict_group_head(shared_task_features),
            "conflict17_logits": self.conflict17_head(conflict17_features),
            "start_pred": self.start_head(fused).squeeze(1),
        }



# ============================================================
# DATA PREP
# ============================================================
# Locate a video file for a BDD_ID by checking common extensions/capitalization.
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


# Load/clean conflict labels from bdd_sce.csv into one normalized row per BDD_ID
def load_conflict_lookup() -> pd.DataFrame:
    if not BDD_SCE_PATH.exists():
        return pd.DataFrame(columns=["BDD_ID", "CONFLICT_TYPE_NORM"])
    df = pd.read_csv(BDD_SCE_PATH).copy()
    if "BDD_ID" not in df.columns:
        return pd.DataFrame(columns=["BDD_ID", "CONFLICT_TYPE_NORM"])
    conflict_col = "CONFLICT_TYPE" if "CONFLICT_TYPE" in df.columns else ("CONFLICT_T" if "CONFLICT_T" in df.columns else None)
    if conflict_col is None:
        return pd.DataFrame(columns=["BDD_ID", "CONFLICT_TYPE_NORM"])
    df["BDD_ID"] = df["BDD_ID"].astype(str)
    df["CONFLICT_TYPE_NORM"] = df[conflict_col].map(safe_upper_conflict_code)
    df = df.sort_values(by=["BDD_ID", "CONFLICT_TYPE_NORM"], na_position="last")
    df = df.drop_duplicates(subset=["BDD_ID"], keep="first")
    return df[["BDD_ID", "CONFLICT_TYPE_NORM"]]


# Construct the master dataframe containing metadata, video paths, start times, and all labels
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

    conflict_lookup = load_conflict_lookup()
    if len(conflict_lookup):
        meta = meta.merge(conflict_lookup, on="BDD_ID", how="left")
    else:
        meta["CONFLICT_TYPE_NORM"] = None

    meta["BDD_START"] = meta["BDD_START"].fillna(0.0).astype(float).clip(lower=0.0)
    meta["conflict_group_target_idx"] = meta["CONFLICT_TYPE_NORM"].map(
        lambda x: GROUP_TO_IDX[CONFLICT_TO_GROUP[x]] if x in CONFLICT_TO_GROUP else IGNORE_CONFLICT_GROUP_INDEX
    )
    meta["conflict17_target_idx"] = meta["CONFLICT_TYPE_NORM"].map(
        lambda x: CONFLICT17_TO_IDX[x] if x in CONFLICT17_TO_IDX else IGNORE_CONFLICT_17_INDEX
    )
    meta.loc[meta["target_idx"] != 0, "conflict_group_target_idx"] = IGNORE_CONFLICT_GROUP_INDEX
    meta.loc[meta["target_idx"] != 0, "conflict17_target_idx"] = IGNORE_CONFLICT_17_INDEX

    df = meta.dropna(subset=["video_path"]).drop_duplicates(subset=["BDD_ID"]).reset_index(drop=True)
    if len(df) == 0:
        raise ValueError(f"No matching videos found in {VIDEO_ROOT} for BDD_ID values in meta.csv")
    return df


# Split train/val/test by BDD_ID groups to avoid leakage across splits
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


# Standardize time-series channels using training split statistics only
def normalize_time_series_train_only(X_ts_sel: np.ndarray, train_mask: np.ndarray) -> np.ndarray:
    if X_ts_sel.ndim != 3:
        raise ValueError(f"Expected X_ts shape [N, C, T], got {X_ts_sel.shape}")
    n, c, t = X_ts_sel.shape
    x_train = X_ts_sel[train_mask].transpose(0, 2, 1).reshape(-1, c)
    scaler = StandardScaler()
    scaler.fit(x_train)
    x_all = X_ts_sel.transpose(0, 2, 1).reshape(-1, c)
    x_all = scaler.transform(x_all)
    return x_all.reshape(n, t, c).transpose(0, 2, 1).astype(np.float32)


# Load arrays/tables, align them by row index, scale features, save splits, and return model input
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
    X_ts_sel = normalize_time_series_train_only(X_ts_sel, train_mask)
    ctx_scaler = StandardScaler()
    X_ctx_sel[train_mask] = ctx_scaler.fit_transform(X_ctx_sel[train_mask])
    X_ctx_sel[~train_mask] = ctx_scaler.transform(X_ctx_sel[~train_mask])
    df.to_csv(SPLITS_PATH, index=False)
    print(f"Saved fresh split metadata to: {SPLITS_PATH}")
    return df, X_ts_sel, X_ctx_sel

# ============================================================
# VIDEO PREPROCESS
# ============================================================
# Decode, sample, resize/crop, format, and normalize videos for SlowFast
# Callable video loader used inside the Dataset for each sample
class SimpleSlowFastVideoLoader:
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
        return self._normalize_video_tensor(packed["inputs"].float())


# ============================================================
# DATASET
# ============================================================
# Returns one training/eval sample: video, kinematics, context, labels, start time, and BDD_ID
class JointFusionDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, X_ts: np.ndarray, X_ctx: np.ndarray, video_loader: SimpleSlowFastVideoLoader) -> None:
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
        return (
            self.video_loader(video_path, bdd_start),
            self.X_ts[idx],
            self.X_ctx[idx],
            int(row["target_idx"]),
            int(row["conflict_group_target_idx"]),
            int(row["conflict17_target_idx"]),
            float(bdd_start),
            str(row["BDD_ID"]),
        )


# Custom batch builder that stacks tensors and converts labels into tensors.
def joint_collate_fn(batch):
    videos, xs_ts, xs_ctx, event_targets, conflict_group_targets, conflict17_targets, start_targets, ids = zip(*batch)
    return (
        torch.stack(videos, dim=0),
        torch.stack(xs_ts, dim=0),
        torch.stack(xs_ctx, dim=0),
        torch.tensor(event_targets, dtype=torch.long),
        torch.tensor(conflict_group_targets, dtype=torch.long),
        torch.tensor(conflict17_targets, dtype=torch.long),
        torch.tensor(start_targets, dtype=torch.float32),
        list(ids),
    )

# ============================================================
# TRAIN / EVAL
# ============================================================
# DataLoaders, optimizer/scheduler, evaluation, training loop, and test reporting.
# Create train, validation, and test DataLoaders from the split dataframe.
def make_loaders(df: pd.DataFrame, X_ts: np.ndarray, X_ctx: np.ndarray):
    train_df = df[df["split"] == "train"].copy()
    val_df = df[df["split"] == "val"].copy()
    test_df = df[df["split"] == "test"].copy()

    video_loader = SimpleSlowFastVideoLoader()
    effective_workers = min(NUM_WORKERS, max(get_visible_cpu_count() - 1, 0))
    persistent_workers = effective_workers > 0
    loader_kwargs = dict(
        num_workers=effective_workers,
        pin_memory=PIN_MEMORY,
        collate_fn=joint_collate_fn,
        persistent_workers=persistent_workers,
    )
    if effective_workers > 0:
        loader_kwargs["prefetch_factor"] = 2

    print(f"DataLoader workers: {effective_workers} | visible CPUs: {get_visible_cpu_count()} | pin_memory: {PIN_MEMORY}")
    train_loader = DataLoader(
        JointFusionDataset(train_df, X_ts[df["split"] == "train"], X_ctx[df["split"] == "train"], video_loader),
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        JointFusionDataset(val_df, X_ts[df["split"] == "val"], X_ctx[df["split"] == "val"], video_loader),
        batch_size=BATCH_SIZE,
        shuffle=False,
        **loader_kwargs,
    )
    test_loader = DataLoader(
        JointFusionDataset(test_df, X_ts[df["split"] == "test"], X_ctx[df["split"] == "test"], video_loader),
        batch_size=BATCH_SIZE,
        shuffle=False,
        **loader_kwargs,
    )
    return train_loader, val_loader, test_loader


# Optional controls to freeze all or part of the SlowFast video backbone.
def maybe_freeze_partial(model: nn.Module) -> None:
    if FREEZE_VIDEO_BACKBONE:
        for p in model.video_encoder.backbone.parameters():
            p.requires_grad = False
        return
    if UNFREEZE_STAGE is not None:
        for name, p in model.video_encoder.backbone.named_parameters():
            p.requires_grad = UNFREEZE_STAGE in name


# AdamW optimizer with separate learning rates for pretrained backbone vs. new heads/fusion.
def build_optimizer_and_scheduler(model: nn.Module, total_steps: int):
    backbone_params, head_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (backbone_params if name.startswith("video_encoder.backbone") else head_params).append(p)
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

    return optimizer, torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


# Compute inverse-frequency class weights to reduce imbalance effects.
def compute_class_weights_from_targets(targets: np.ndarray, num_classes: int) -> torch.Tensor:
    counts = np.bincount(targets, minlength=num_classes)
    counts = np.maximum(counts, 1)
    weights = len(targets) / (num_classes * counts) if len(targets) else np.ones(num_classes)
    return torch.tensor(weights, dtype=torch.float32, device=DEVICE)


# Run model inference over a loader and collect predictions, probabilities, and metrics.
def evaluate_loader(model: nn.Module, loader: DataLoader) -> Dict[str, object]:
    model.eval()
    event_true: List[int] = []
    event_pred: List[int] = []
    conflict_group_true: List[int] = []
    conflict_group_pred: List[int] = []
    conflict17_true: List[int] = []
    conflict17_pred: List[int] = []
    start_true: List[float] = []
    start_pred: List[float] = []
    ids: List[str] = []
    event_prob_chunks: List[np.ndarray] = []
    conflict_group_prob_chunks: List[np.ndarray] = []
    conflict17_prob_chunks: List[np.ndarray] = []

    if DEVICE.type == "cuda":
        torch.cuda.reset_peak_memory_stats(DEVICE)
    eval_start = time.perf_counter()

    with torch.no_grad():
        for batch_idx, (batch_v, batch_ts, batch_ctx, batch_evt, batch_group, batch_c17, batch_start, batch_ids) in enumerate(
            tqdm(loader, desc="Evaluating", leave=False), start=1
        ):
            batch_time_start = time.perf_counter()
            batch_v = batch_v.to(DEVICE, non_blocking=PIN_MEMORY)
            batch_ts = batch_ts.to(DEVICE, non_blocking=PIN_MEMORY)
            batch_ctx = batch_ctx.to(DEVICE, non_blocking=PIN_MEMORY)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=USE_AMP):
                outputs = model(batch_v, batch_ts, batch_ctx)
                event_probs = torch.softmax(outputs["event_logits"], dim=1)
                conflict_group_probs = torch.softmax(outputs["conflict_group_logits"], dim=1)
                conflict17_probs = torch.softmax(outputs["conflict17_logits"], dim=1)
            if DEVICE.type == "cuda":
                torch.cuda.synchronize(DEVICE)
            batch_time = time.perf_counter() - batch_time_start

            evt_p = torch.argmax(outputs["event_logits"], dim=1).cpu().numpy()
            group_p = torch.argmax(outputs["conflict_group_logits"], dim=1).cpu().numpy()
            c17_p = torch.argmax(outputs["conflict17_logits"], dim=1).cpu().numpy()
            start_p = outputs["start_pred"].detach().cpu().numpy()

            event_true.extend(batch_evt.numpy().tolist())
            event_pred.extend(evt_p.tolist())
            conflict_group_true.extend(batch_group.numpy().tolist())
            conflict_group_pred.extend(group_p.tolist())
            conflict17_true.extend(batch_c17.numpy().tolist())
            conflict17_pred.extend(c17_p.tolist())
            start_true.extend(batch_start.numpy().tolist())
            start_pred.extend(start_p.tolist())
            ids.extend(list(batch_ids))
            event_prob_chunks.append(event_probs.detach().cpu().numpy())
            conflict_group_prob_chunks.append(conflict_group_probs.detach().cpu().numpy())
            conflict17_prob_chunks.append(conflict17_probs.detach().cpu().numpy())

            if batch_idx % GPU_LOG_INTERVAL == 0 or batch_idx == len(loader):
                stats = get_gpu_stats()
                print(
                    f"Eval batch {batch_idx:03d}/{len(loader):03d} | batch time: {batch_time:.2f}s | "
                    f"GPU alloc: {stats['allocated']} | GPU reserved: {stats['reserved']} | peak alloc: {stats['max_allocated']}"
                )

    total_time = time.perf_counter() - eval_start
    print_gpu_stats(f"Evaluation complete in {total_time:.2f}s")

    group_mask = [(et == 0) and (cg >= 0) for et, cg in zip(event_true, conflict_group_true)]
    if any(group_mask):
        cg_true = [c for c, m in zip(conflict_group_true, group_mask) if m]
        cg_pred = [c for c, m in zip(conflict_group_pred, group_mask) if m]
        conflict_group_acc = accuracy_score(cg_true, cg_pred)
    else:
        conflict_group_acc = float("nan")

    c17_mask = [(et == 0) and (c17 >= 0) for et, c17 in zip(event_true, conflict17_true)]
    if any(c17_mask):
        c17_true_valid = [c for c, m in zip(conflict17_true, c17_mask) if m]
        c17_pred_valid = [c for c, m in zip(conflict17_pred, c17_mask) if m]
        conflict17_acc = accuracy_score(c17_true_valid, c17_pred_valid)
    else:
        conflict17_acc = float("nan")

    return {
        "event_acc": accuracy_score(event_true, event_pred),
        "conflict_group_acc": conflict_group_acc,
        "conflict17_acc": conflict17_acc,
        "start_mae": float(mean_absolute_error(start_true, start_pred)),
        "start_rmse": float(np.sqrt(mean_squared_error(start_true, start_pred))),
        "event_true": event_true,
        "event_pred": event_pred,
        "conflict_group_true": conflict_group_true,
        "conflict_group_pred": conflict_group_pred,
        "conflict17_true": conflict17_true,
        "conflict17_pred": conflict17_pred,
        "start_true": start_true,
        "start_pred": start_pred,
        "ids": ids,
        "event_probs": np.concatenate(event_prob_chunks, axis=0) if event_prob_chunks else np.zeros((0, NUM_EVENT_CLASSES), dtype=np.float32),
        "conflict_group_probs": np.concatenate(conflict_group_prob_chunks, axis=0) if conflict_group_prob_chunks else np.zeros((0, NUM_CONFLICT_GROUP_CLASSES), dtype=np.float32),
        "conflict17_probs": np.concatenate(conflict17_prob_chunks, axis=0) if conflict17_prob_chunks else np.zeros((0, NUM_CONFLICT_17_CLASSES), dtype=np.float32),
    }


# Train the four-head model, validate each epoch, save best checkpoint, and early stop.
def train_model(df: pd.DataFrame, X_ts: np.ndarray, X_ctx: np.ndarray) -> JointMultiTaskFourHeadModel:
    train_loader, val_loader, _ = make_loaders(df, X_ts, X_ctx)
    video_encoder = MMAction2SlowFastFeatureExtractor(
        config_path=MMACTION_CONFIG_PATH,
        checkpoint_path=MMACTION_CHECKPOINT_PATH,
        freeze_backbone=FREEZE_VIDEO_BACKBONE,
    )
    model = JointMultiTaskFourHeadModel(
        video_encoder=video_encoder,
        in_chans_ts=X_ts.shape[1],
        in_chans_ctx=X_ctx.shape[1],
    ).to(DEVICE)
    maybe_freeze_partial(model)

    train_events = df[df["split"] == "train"]["target_idx"].to_numpy(dtype=int)
    event_weights = compute_class_weights_from_targets(train_events, NUM_EVENT_CLASSES)

    train_group = df[(df["split"] == "train") & (df["target_idx"] == 0) & (df["conflict_group_target_idx"] >= 0)]["conflict_group_target_idx"].to_numpy(dtype=int)
    group_weights = compute_class_weights_from_targets(train_group, NUM_CONFLICT_GROUP_CLASSES)

    train_c17 = df[(df["split"] == "train") & (df["target_idx"] == 0) & (df["conflict17_target_idx"] >= 0)]["conflict17_target_idx"].to_numpy(dtype=int)
    c17_weights = compute_class_weights_from_targets(train_c17, NUM_CONFLICT_17_CLASSES)

    event_criterion = FocalLoss(alpha=event_weights, gamma=FOCAL_GAMMA)
    conflict_group_criterion = nn.CrossEntropyLoss(weight=group_weights)
    conflict17_criterion = nn.CrossEntropyLoss(weight=c17_weights)
    start_criterion = nn.SmoothL1Loss()

    optimizer, scheduler = build_optimizer_and_scheduler(model, total_steps=len(train_loader))
    scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP)

    best_val_score = -1.0
    patience_ctr = 0
    history: List[Dict[str, float]] = []

    for epoch in range(EPOCHS):
        model.train()
        running_total = running_event = running_group = running_c17 = running_start = 0.0
        event_correct = event_total = 0
        group_correct = group_total = 0
        c17_correct = c17_total = 0
        epoch_start = time.perf_counter()
        if DEVICE.type == "cuda":
            torch.cuda.reset_peak_memory_stats(DEVICE)

        for batch_idx, (batch_v, batch_ts, batch_ctx, batch_evt, batch_group, batch_c17, batch_start, _) in enumerate(
            tqdm(train_loader, desc=f"Epoch {epoch + 1:02d}/{EPOCHS}", leave=False), start=1
        ):
            batch_time_start = time.perf_counter()
            batch_v = batch_v.to(DEVICE, non_blocking=PIN_MEMORY)
            batch_ts = batch_ts.to(DEVICE, non_blocking=PIN_MEMORY)
            batch_ctx = batch_ctx.to(DEVICE, non_blocking=PIN_MEMORY)
            batch_evt = batch_evt.to(DEVICE, non_blocking=PIN_MEMORY)
            batch_group = batch_group.to(DEVICE, non_blocking=PIN_MEMORY)
            batch_c17 = batch_c17.to(DEVICE, non_blocking=PIN_MEMORY)
            batch_start = batch_start.to(DEVICE, non_blocking=PIN_MEMORY)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=USE_AMP):
                outputs = model(batch_v, batch_ts, batch_ctx)
                loss, loss_info = compute_multitask_loss(
                    outputs=outputs,
                    event_targets=batch_evt,
                    conflict_group_targets=batch_group,
                    conflict17_targets=batch_c17,
                    start_targets=batch_start,
                    event_criterion=event_criterion,
                    conflict_group_criterion=conflict_group_criterion,
                    conflict17_criterion=conflict17_criterion,
                    start_criterion=start_criterion,
                    epoch=epoch,
                )
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            if DEVICE.type == "cuda":
                torch.cuda.synchronize(DEVICE)
            batch_time = time.perf_counter() - batch_time_start

            bs = batch_evt.size(0)
            running_total += loss_info["total"] * bs
            running_event += loss_info["event"] * bs
            running_start += loss_info["start"] * bs
            if loss_info["n_conflict_group"] > 0:
                running_group += loss_info["conflict_group"] * loss_info["n_conflict_group"]
            if loss_info["n_conflict17"] > 0:
                running_c17 += loss_info["conflict17"] * loss_info["n_conflict17"]

            evt_preds = torch.argmax(outputs["event_logits"], dim=1)
            event_correct += (evt_preds == batch_evt).sum().item()
            event_total += bs

            group_mask = (batch_evt == 0) & (batch_group >= 0)
            if group_mask.any():
                group_preds = torch.argmax(outputs["conflict_group_logits"][group_mask], dim=1)
                group_correct += (group_preds == batch_group[group_mask]).sum().item()
                group_total += int(group_mask.sum().item())

            c17_mask = (batch_evt == 0) & (batch_c17 >= 0)
            if c17_mask.any():
                c17_preds = torch.argmax(outputs["conflict17_logits"][c17_mask], dim=1)
                c17_correct += (c17_preds == batch_c17[c17_mask]).sum().item()
                c17_total += int(c17_mask.sum().item())

            if batch_idx % GPU_LOG_INTERVAL == 0 or batch_idx == len(train_loader):
                stats = get_gpu_stats()
                print(
                    f"Epoch {epoch + 1:02d} batch {batch_idx:03d}/{len(train_loader):03d} | batch time: {batch_time:.2f}s | "
                    f"samples: {bs} | L_total: {loss_info['total']:.4f} | L_evt: {loss_info['event']:.4f} | "
                    f"L_grp: {loss_info['conflict_group']:.4f} (n={loss_info['n_conflict_group']}, w={loss_info['current_group_weight']:.2f}) | "
                    f"L_c17: {loss_info['conflict17']:.4f} (n={loss_info['n_conflict17']}, w={loss_info['current_c17_weight']:.2f}, detach={DETACH_CONFLICT17_HEAD}) | "
                    f"L_start: {loss_info['start']:.4f} | backbone_lr: {optimizer.param_groups[0]['lr']:.2e} | "
                    f"head_lr: {optimizer.param_groups[1]['lr']:.2e} | GPU alloc: {stats['allocated']} | peak alloc: {stats['max_allocated']}"
                )

        train_total_loss = running_total / max(event_total, 1)
        train_event_acc = event_correct / max(event_total, 1)
        train_group_acc = group_correct / max(group_total, 1) if group_total > 0 else float("nan")
        train_c17_acc = c17_correct / max(c17_total, 1) if c17_total > 0 else float("nan")

        epoch_time = time.perf_counter() - epoch_start
        samples_per_sec = event_total / max(epoch_time, 1e-8)
        val_metrics = evaluate_loader(model, val_loader)
        val_event_acc = val_metrics["event_acc"]
        val_group_acc = val_metrics["conflict_group_acc"]
        val_c17_acc = val_metrics["conflict17_acc"]
        val_start_mae = val_metrics["start_mae"]

        val_score = (
            0.68 * val_event_acc
            + 0.18 * (0.0 if math.isnan(val_group_acc) else val_group_acc)
            + 0.06 * (0.0 if math.isnan(val_c17_acc) else val_c17_acc)
            + 0.08 * (1.0 / (1.0 + val_start_mae))
        )

        history.append({
            "epoch": epoch + 1,
            "train_total_loss": train_total_loss,
            "train_event_acc": train_event_acc,
            "train_group_acc": train_group_acc,
            "train_c17_acc": train_c17_acc,
            "val_event_acc": val_event_acc,
            "val_group_acc": val_group_acc,
            "val_c17_acc": val_c17_acc,
            "val_start_mae": val_start_mae,
            "val_score": val_score,
        })
        print(
            f"Epoch {epoch + 1:02d} | Train L_total: {train_total_loss:.4f} | Train Evt Acc: {train_event_acc:.4f} | "
            f"Train Group Acc: {train_group_acc:.4f} | Train 17-way Acc: {train_c17_acc:.4f} | "
            f"Val Evt Acc: {val_event_acc:.4f} | Val Group Acc: {val_group_acc:.4f} | Val 17-way Acc: {val_c17_acc:.4f} | "
            f"Val Start MAE: {val_start_mae:.3f}s | Val Score: {val_score:.4f} | Epoch Time: {epoch_time:.2f}s | Throughput: {samples_per_sec:.2f} samples/s"
        )
        print_gpu_stats(f"Epoch {epoch + 1:02d} GPU summary")

        if val_score > best_val_score:
            best_val_score = val_score
            patience_ctr = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "history": history,
                "video_dim": model.video_encoder.out_dim,
            }, MODEL_PATH)
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                print(f"Early stopping at epoch {epoch + 1}.")
                break

    print(f"\nBest joint multi-task four-head model saved to: {MODEL_PATH}")
    return model


# Evaluate the saved best model on the held-out test split and write CSV/JSON outputs
def evaluate_test(df: pd.DataFrame, X_ts: np.ndarray, X_ctx: np.ndarray) -> None:
    _, _, test_loader = make_loaders(df, X_ts, X_ctx)
    video_encoder = MMAction2SlowFastFeatureExtractor(
        config_path=MMACTION_CONFIG_PATH,
        checkpoint_path=MMACTION_CHECKPOINT_PATH,
        freeze_backbone=False,
    )
    model = JointMultiTaskFourHeadModel(
        video_encoder=video_encoder,
        in_chans_ts=X_ts.shape[1],
        in_chans_ctx=X_ctx.shape[1],
    ).to(DEVICE)
    ckpt = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    metrics = evaluate_loader(model, test_loader)
    event_true = metrics["event_true"]
    event_pred = metrics["event_pred"]
    group_true = metrics["conflict_group_true"]
    group_pred = metrics["conflict_group_pred"]
    c17_true = metrics["conflict17_true"]
    c17_pred = metrics["conflict17_pred"]
    start_true = metrics["start_true"]
    start_pred = metrics["start_pred"]
    ids = metrics["ids"]
    event_probs = metrics["event_probs"]
    group_probs = metrics["conflict_group_probs"]
    c17_probs = metrics["conflict17_probs"]

    print("\n" + "=" * 80)
    print("JOINT MULTI-TASK FOUR-HEAD TEST RESULTS")
    print(f"[Event Type] Accuracy: {metrics['event_acc']:.4f}")
    event_report = classification_report(
        event_true,
        event_pred,
        target_names=[CLASS_MAP_STR[i] for i in range(NUM_EVENT_CLASSES)],
        output_dict=True,
        zero_division=0,
    )
    print(classification_report(
        event_true,
        event_pred,
        target_names=[CLASS_MAP_STR[i] for i in range(NUM_EVENT_CLASSES)],
        zero_division=0,
    ))

    group_mask = [(et == 0) and (cg >= 0) for et, cg in zip(event_true, group_true)]
    if any(group_mask):
        gt = [c for c, m in zip(group_true, group_mask) if m]
        gp = [c for c, m in zip(group_pred, group_mask) if m]
        print(f"[Conflict Group] Accuracy on {len(gt)} conflict samples: {metrics['conflict_group_acc']:.4f}")
        group_report = classification_report(
            gt,
            gp,
            labels=list(range(NUM_CONFLICT_GROUP_CLASSES)),
            target_names=GROUP_NAMES,
            output_dict=True,
            zero_division=0,
        )
        print(classification_report(
            gt,
            gp,
            labels=list(range(NUM_CONFLICT_GROUP_CLASSES)),
            target_names=GROUP_NAMES,
            zero_division=0,
        ))
    else:
        group_report = {}
        print("[Conflict Group] No conflict samples with valid group labels in test set.")

    c17_mask = [(et == 0) and (c17 >= 0) for et, c17 in zip(event_true, c17_true)]
    if any(c17_mask):
        c17t = [c for c, m in zip(c17_true, c17_mask) if m]
        c17p = [c for c, m in zip(c17_pred, c17_mask) if m]
        print(f"[Conflict 17-way] Accuracy on {len(c17t)} conflict samples: {metrics['conflict17_acc']:.4f}")
        c17_report = classification_report(
            c17t,
            c17p,
            labels=list(range(NUM_CONFLICT_17_CLASSES)),
            target_names=CONFLICT_TYPES_17,
            output_dict=True,
            zero_division=0,
        )
    else:
        c17_report = {}
        print("[Conflict 17-way] No conflict samples with valid 17-way labels in test set.")

    print(f"[Start Time] MAE: {metrics['start_mae']:.3f}s | RMSE: {metrics['start_rmse']:.3f}s")

    out = pd.DataFrame({
        "BDD_ID": ids,
        "event_true_idx": event_true,
        "event_pred_idx": event_pred,
        "event_true_label": [CLASS_MAP_STR[i] for i in event_true],
        "event_pred_label": [CLASS_MAP_STR[i] for i in event_pred],
        "conflict_group_true": [IDX_TO_GROUP[c] if (et == 0 and c >= 0) else "" for et, c in zip(event_true, group_true)],
        "conflict_group_pred": [IDX_TO_GROUP[c] if ep == 0 else "" for ep, c in zip(event_pred, group_pred)],
        "conflict17_true": [IDX_TO_CONFLICT17[c] if (et == 0 and c >= 0) else "" for et, c in zip(event_true, c17_true)],
        "conflict17_pred": [IDX_TO_CONFLICT17[c] if ep == 0 else "" for ep, c in zip(event_pred, c17_pred)],
        "start_true": start_true,
        "start_pred": start_pred,
        "start_abs_error": [abs(a - b) for a, b in zip(start_true, start_pred)],
        "Conflict": event_probs[:, 0] if len(event_probs) else [],
        "Bump": event_probs[:, 1] if len(event_probs) else [],
        "Hard Brake": event_probs[:, 2] if len(event_probs) else [],
        "Not SCE": event_probs[:, 3] if len(event_probs) else [],
    })
    for i, group_name in enumerate(GROUP_NAMES):
        out[f"confgrp_{group_name}"] = group_probs[:, i] if len(group_probs) else []
    for i, c17_name in enumerate(CONFLICT_TYPES_17):
        out[f"conf17_{c17_name}"] = c17_probs[:, i] if len(c17_probs) else []
    out.to_csv(PRED_PATH, index=False)

    with open(METRICS_PATH, "w") as f:
        json.dump({
            "event": {"accuracy": metrics["event_acc"], "classification_report": event_report},
            "conflict_group": {"accuracy": metrics["conflict_group_acc"], "classification_report": group_report},
            "conflict_17way": {"accuracy": metrics["conflict17_acc"], "classification_report": c17_report},
            "start_time": {"mae_seconds": metrics["start_mae"], "rmse_seconds": metrics["start_rmse"]},
            "config": {
                "clip_len": CLIP_LEN,
                "frame_interval": FRAME_INTERVAL,
                "num_workers": NUM_WORKERS,
                "lambda_event": LAMBDA_EVENT,
                "lambda_conflict_group": LAMBDA_CONFLICT_GROUP,
                "lambda_conflict_17": LAMBDA_CONFLICT_17,
                "lambda_start": LAMBDA_START,
                "aux_warmup_epochs": AUX_WARMUP_EPOCHS,
                "detach_conflict17_head": DETACH_CONFLICT17_HEAD,
            },
        }, f, indent=2)

    print(f"\nSaved predictions to: {PRED_PATH}")
    print(f"Saved metrics to: {METRICS_PATH}")
    print("=" * 80)

# ============================================================
# MAIN
# ============================================================
# Main script execution function.
def main() -> None:
    register_all_modules()
    set_seed(SEED)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    print(f"Visible CPUs for this job: {get_visible_cpu_count()}")
    print(f"Configured DataLoader workers: {NUM_WORKERS}")
    if get_visible_cpu_count() <= 1:
        print("WARNING: Only 1 CPU visible to the job. Video decode/preprocess may bottleneck GPU utilization.")
    df, X_ts, X_ctx = load_and_align_data()
    train_model(df, X_ts, X_ctx)
    evaluate_test(df, X_ts, X_ctx)


if __name__ == "__main__":
    main()
