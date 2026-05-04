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

MMACTION_CONFIG_PATH = Path("/home/adityak22/mmaction2/configs/recognition/slowfast/slowfast_r50_8xb8-4x16x1-256e_kinetics400-rgb.py")
MMACTION_CHECKPOINT_PATH = Path("/home/adityak22/mmaction2/checkpoints/slowfast_r50_8xb8-4x16x1-256e_kinetics400-rgb_20220901-701b0f6f.pth")

OUT_DIR = Path("data/mmaction2_slowfast_joint_multitask_fusion")
OUT_DIR.mkdir(parents=True, exist_ok=True)
SPLITS_PATH = OUT_DIR / "joint_splits.csv"
MODEL_PATH = OUT_DIR / "joint_multitask_fusion_best.pth"
PRED_PATH = OUT_DIR / "joint_multitask_test_predictions.csv"
METRICS_PATH = OUT_DIR / "joint_multitask_test_metrics.json"

BATCH_SIZE = 4
EPOCHS = 20
PATIENCE = 6
LEARNING_RATE = 1e-4
BACKBONE_LEARNING_RATE = 1e-5
WEIGHT_DECAY = 1e-4
SEED = 1
NUM_EVENT_CLASSES = 4
NUM_CONFLICT_CLASSES = 17
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

# Multi-task loss weights. Event is the primary task; conflict subtype is
# secondary but still classification. Start time is regressed directly in
# seconds so its weight is kept small to prevent it from dominating the
# classification losses.
LAMBDA_EVENT = 1.0
LAMBDA_CONFLICT = 0.5
LAMBDA_START = 0.05

CLASS_MAP_STR = {
    0: "Conflict",
    1: "Bump",
    2: "Hard Brake",
    3: "Not an SCE",
}

# CONFLICT_T subtype labels (per metadata.pdf). Alphabetical ordering keeps
# the mapping reproducible across runs. -1 is used as a sentinel for samples
# whose EVENT_TYPE is not Conflict; those rows are masked out of the subtype
# loss during training and excluded from subtype metrics at eval time.
CONFLICT_T_MAP = {
    "A": 0, "D": 1, "E": 2, "F": 3, "G": 4, "H": 5, "I": 6, "J": 7,
    "O": 8, "P": 9, "Q": 10, "R": 11, "S": 12, "T": 13, "U": 14, "W": 15, "Y": 16,
}
CONFLICT_T_INV = {v: k for k, v in CONFLICT_T_MAP.items()}

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


def compute_multitask_loss(
    outputs: Dict[str, torch.Tensor],
    event_targets: torch.Tensor,
    conflict_targets: torch.Tensor,
    start_targets: torch.Tensor,
    event_criterion: nn.Module,
    conflict_criterion: nn.Module,
    start_criterion: nn.Module,
    lambda_event: float = LAMBDA_EVENT,
    lambda_conflict: float = LAMBDA_CONFLICT,
    lambda_start: float = LAMBDA_START,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Combined loss for event / conflict subtype / start time regression.

    The conflict head is only trained on samples that are actually conflicts
    (event_target == 0) AND have a valid subtype label (conflict_target >= 0).
    Other rows are masked out of the conflict loss entirely but still contribute
    to the event and start-time losses.
    """
    event_loss = event_criterion(outputs["event_logits"], event_targets)
    start_loss = start_criterion(outputs["start_pred"], start_targets)

    conflict_mask = (event_targets == 0) & (conflict_targets >= 0)
    if conflict_mask.any():
        conflict_loss = conflict_criterion(
            outputs["conflict_logits"][conflict_mask],
            conflict_targets[conflict_mask],
        )
    else:
        conflict_loss = torch.zeros((), device=event_loss.device)

    total = lambda_event * event_loss + lambda_conflict * conflict_loss + lambda_start * start_loss

    loss_dict = {
        "total": total.item(),
        "event": event_loss.item(),
        "conflict": conflict_loss.item(),
        "start": start_loss.item(),
        "n_conflict": int(conflict_mask.sum().item()),
    }
    return total, loss_dict


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
# JOINT MULTI-TASK MODEL
# =========================================================
class JointEndToEndFusionModel(nn.Module):
    """Shared fusion backbone with three task-specific heads:

    - event_head: 4-class classifier (Conflict / Bump / Hard Brake / Not an SCE)
    - conflict_head: 17-class classifier for conflict subtype (CONFLICT_T)
    - start_head: regression head for BDD_START in seconds

    The fusion trunk ends at a 256-d shared representation that all three
    heads read from independently.
    """

    def __init__(
        self,
        video_encoder: MMAction2SlowFastFeatureExtractor,
        in_chans_ts: int,
        in_chans_ctx: int,
        num_event_classes: int = NUM_EVENT_CLASSES,
        num_conflict_classes: int = NUM_CONFLICT_CLASSES,
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
        # Shared fusion trunk. Outputs 256-d; heads take over from here.
        self.fusion = nn.Sequential(
            nn.Linear(256 * 4, 512),
            nn.ReLU(),
            nn.Dropout(0.35),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.25),
        )
        self.event_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_event_classes),
        )
        self.conflict_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_conflict_classes),
        )
        self.start_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1),
        )

    def forward(
        self,
        video_inputs: torch.Tensor,
        x_ts: torch.Tensor,
        x_ctx: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        video_embedding = self.video_encoder(video_inputs)
        kine_embedding = self.kine_encoder(x_ts, x_ctx)

        v = self.video_proj(video_embedding)
        k = self.kine_proj(kine_embedding)
        joint = torch.cat([v, k, v * k, torch.abs(v - k)], dim=1)
        shared = self.fusion(joint)

        return {
            "event_logits": self.event_head(shared),
            "conflict_logits": self.conflict_head(shared),
            "start_pred": self.start_head(shared).squeeze(1),
        }


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


def _map_conflict_t(value) -> int:
    """Map a raw CONFLICT_T letter to its integer class index, -1 if missing."""
    if pd.isna(value):
        return -1
    code = str(value).strip().upper()
    return CONFLICT_T_MAP.get(code, -1)


def build_base_dataframe() -> pd.DataFrame:
    meta = pd.read_csv(KINE_META_PATH).copy()
    if "BDD_ID" not in meta.columns or "y" not in meta.columns:
        raise ValueError("meta.csv must contain BDD_ID and y.")

    meta["BDD_ID"] = meta["BDD_ID"].astype(str)
    meta["row_idx"] = np.arange(len(meta))
    meta["target_idx"] = meta["y"].astype(int)
    meta["video_path"] = meta["BDD_ID"].map(build_video_path)

    # Defaults that hold whether or not bdd_sce.csv is available.
    meta["BDD_START"] = 0.0
    meta["conflict_idx"] = -1

    if BDD_SCE_PATH.exists():
        bdd_sce = pd.read_csv(BDD_SCE_PATH).copy()
        if "BDD_ID" in bdd_sce.columns:
            bdd_sce["BDD_ID"] = bdd_sce["BDD_ID"].astype(str)
            keep_cols = ["BDD_ID"]
            if "BDD_START" in bdd_sce.columns:
                keep_cols.append("BDD_START")
            if "CONFLICT_T" in bdd_sce.columns:
                keep_cols.append("CONFLICT_T")

            bdd_sce = (
                bdd_sce[keep_cols]
                .dropna(subset=["BDD_ID"])
                .drop_duplicates(subset=["BDD_ID"], keep="first")
            )

            # Drop the placeholder columns that we set above so the merge
            # brings in the real values cleanly.
            drop_cols = [c for c in ["BDD_START", "CONFLICT_T"] if c in keep_cols and c in meta.columns]
            meta = meta.drop(columns=drop_cols)
            meta = meta.merge(bdd_sce, on="BDD_ID", how="left")

            if "CONFLICT_T" in meta.columns:
                meta["conflict_idx"] = meta["CONFLICT_T"].map(_map_conflict_t).astype(int)
    else:
        print(
            f"[warn] {BDD_SCE_PATH} not found. Conflict subtype head will "
            "train on 0 samples and start time targets will default to 0.0."
        )

    meta["BDD_START"] = meta["BDD_START"].fillna(0.0).astype(float).clip(lower=0.0)
    meta["conflict_idx"] = meta["conflict_idx"].fillna(-1).astype(int)

    df = meta.dropna(subset=["video_path"]).drop_duplicates(subset=["BDD_ID"]).reset_index(drop=True)
    if len(df) == 0:
        raise ValueError(f"No matching videos found in {VIDEO_ROOT} for BDD_ID values in meta.csv")

    n_conflict_rows = int((df["target_idx"] == 0).sum())
    n_conflict_with_subtype = int(((df["target_idx"] == 0) & (df["conflict_idx"] >= 0)).sum())
    print(
        f"[data] total rows: {len(df)} | conflict rows: {n_conflict_rows} | "
        f"conflict rows with subtype label: {n_conflict_with_subtype}"
    )

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

    X_ts_sel = normalize_time_series_train_only(X_ts_sel, train_mask)

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
        event_target = int(row["target_idx"])
        conflict_target = int(row.get("conflict_idx", -1))
        start_target = float(bdd_start)
        bdd_id = str(row["BDD_ID"])
        return video_inputs, x_ts, x_ctx, event_target, conflict_target, start_target, bdd_id


def joint_collate_fn(batch):
    videos, xs_ts, xs_ctx, event_targets, conflict_targets, start_targets, ids = zip(*batch)
    videos = torch.stack(videos, dim=0)
    xs_ts = torch.stack(xs_ts, dim=0)
    xs_ctx = torch.stack(xs_ctx, dim=0)
    event_targets = torch.tensor(event_targets, dtype=torch.long)
    conflict_targets = torch.tensor(conflict_targets, dtype=torch.long)
    start_targets = torch.tensor(start_targets, dtype=torch.float32)
    return videos, xs_ts, xs_ctx, event_targets, conflict_targets, start_targets, list(ids)


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


def evaluate_loader(model: nn.Module, loader: DataLoader) -> Dict[str, object]:
    """Run the multi-head model across a loader and collect predictions.

    Returns a dict with raw lists of targets and predictions for each task
    plus summary metrics. Conflict subtype accuracy is computed only over
    samples whose ground truth event is Conflict AND have a valid subtype
    label.
    """
    model.eval()
    event_true: List[int] = []
    event_pred: List[int] = []
    conflict_true: List[int] = []
    conflict_pred: List[int] = []
    start_true: List[float] = []
    start_pred: List[float] = []
    ids: List[str] = []
    eval_start = time.perf_counter()

    if DEVICE.type == "cuda":
        torch.cuda.reset_peak_memory_stats(DEVICE)

    with torch.no_grad():
        for batch_idx, (batch_v, batch_ts, batch_ctx, batch_evt, batch_conf, batch_start, batch_ids) in enumerate(
            tqdm(loader, desc="Evaluating", leave=False), start=1
        ):
            batch_time_start = time.perf_counter()
            batch_v = batch_v.to(DEVICE)
            batch_ts = batch_ts.to(DEVICE)
            batch_ctx = batch_ctx.to(DEVICE)
            outputs = model(batch_v, batch_ts, batch_ctx)
            if DEVICE.type == "cuda":
                torch.cuda.synchronize(DEVICE)
            batch_time = time.perf_counter() - batch_time_start

            evt_p = torch.argmax(outputs["event_logits"], dim=1).cpu().numpy()
            conf_p = torch.argmax(outputs["conflict_logits"], dim=1).cpu().numpy()
            start_p = outputs["start_pred"].cpu().numpy()

            event_true.extend(batch_evt.numpy().tolist())
            event_pred.extend(evt_p.tolist())
            conflict_true.extend(batch_conf.numpy().tolist())
            conflict_pred.extend(conf_p.tolist())
            start_true.extend(batch_start.numpy().tolist())
            start_pred.extend(start_p.tolist())
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

    event_acc = accuracy_score(event_true, event_pred)

    conflict_eval_mask = [(et == 0) and (ct >= 0) for et, ct in zip(event_true, conflict_true)]
    if any(conflict_eval_mask):
        ct_true = [c for c, m in zip(conflict_true, conflict_eval_mask) if m]
        ct_pred = [c for c, m in zip(conflict_pred, conflict_eval_mask) if m]
        conflict_acc = accuracy_score(ct_true, ct_pred)
    else:
        conflict_acc = float("nan")

    start_mae = float(mean_absolute_error(start_true, start_pred))
    start_rmse = float(np.sqrt(mean_squared_error(start_true, start_pred)))

    return {
        "event_acc": event_acc,
        "conflict_acc": conflict_acc,
        "start_mae": start_mae,
        "start_rmse": start_rmse,
        "event_true": event_true,
        "event_pred": event_pred,
        "conflict_true": conflict_true,
        "conflict_pred": conflict_pred,
        "start_true": start_true,
        "start_pred": start_pred,
        "ids": ids,
    }


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
        num_event_classes=NUM_EVENT_CLASSES,
        num_conflict_classes=NUM_CONFLICT_CLASSES,
    ).to(DEVICE)
    maybe_freeze_partial(model)

    # Event class weights computed from train split (same as old single-head version).
    train_events = df[df["split"] == "train"]["target_idx"].to_numpy(dtype=int)
    event_counts = np.bincount(train_events, minlength=NUM_EVENT_CLASSES)
    event_weights = len(train_events) / (NUM_EVENT_CLASSES * np.maximum(event_counts, 1))
    event_weights = torch.tensor(event_weights, dtype=torch.float32, device=DEVICE)

    # Conflict subtype class weights over train-split conflict rows with labels.
    # If there are no usable rows (e.g., no bdd_sce.csv), fall back to uniform
    # weights so the criterion still constructs.
    train_conf_mask = (df["split"] == "train") & (df["target_idx"] == 0) & (df["conflict_idx"] >= 0)
    train_conflicts = df.loc[train_conf_mask, "conflict_idx"].to_numpy(dtype=int)
    if len(train_conflicts) > 0:
        conf_counts = np.bincount(train_conflicts, minlength=NUM_CONFLICT_CLASSES)
        conf_weights = len(train_conflicts) / (NUM_CONFLICT_CLASSES * np.maximum(conf_counts, 1))
    else:
        conf_weights = np.ones(NUM_CONFLICT_CLASSES, dtype=np.float32)
        print("[warn] No conflict subtype labels found in train split; conflict head is effectively untrained.")
    conf_weights = torch.tensor(conf_weights, dtype=torch.float32, device=DEVICE)

    event_criterion = FocalLoss(alpha=event_weights, gamma=FOCAL_GAMMA)
    conflict_criterion = nn.CrossEntropyLoss(weight=conf_weights)
    start_criterion = nn.SmoothL1Loss()

    optimizer, scheduler = build_optimizer_and_scheduler(model, total_steps=len(train_loader))

    best_val_score = -1.0
    patience_ctr = 0
    history: List[Dict[str, float]] = []
    global_step = 0

    for epoch in range(EPOCHS):
        model.train()
        running_total = 0.0
        running_event = 0.0
        running_conflict = 0.0
        running_start = 0.0
        event_correct = 0
        event_total = 0
        conflict_correct = 0
        conflict_total = 0
        epoch_start = time.perf_counter()

        if DEVICE.type == "cuda":
            torch.cuda.reset_peak_memory_stats(DEVICE)

        for batch_idx, (batch_v, batch_ts, batch_ctx, batch_evt, batch_conf, batch_start, _) in enumerate(
            tqdm(train_loader, desc=f"Epoch {epoch + 1:02d}/{EPOCHS}", leave=False), start=1
        ):
            batch_time_start = time.perf_counter()
            batch_v = batch_v.to(DEVICE)
            batch_ts = batch_ts.to(DEVICE)
            batch_ctx = batch_ctx.to(DEVICE)
            batch_evt = batch_evt.to(DEVICE)
            batch_conf = batch_conf.to(DEVICE)
            batch_start = batch_start.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(batch_v, batch_ts, batch_ctx)
            loss, loss_info = compute_multitask_loss(
                outputs=outputs,
                event_targets=batch_evt,
                conflict_targets=batch_conf,
                start_targets=batch_start,
                event_criterion=event_criterion,
                conflict_criterion=conflict_criterion,
                start_criterion=start_criterion,
            )
            loss.backward()
            optimizer.step()
            scheduler.step()
            global_step += 1

            if DEVICE.type == "cuda":
                torch.cuda.synchronize(DEVICE)
            batch_time = time.perf_counter() - batch_time_start

            bs = batch_evt.size(0)
            running_total += loss_info["total"] * bs
            running_event += loss_info["event"] * bs
            running_start += loss_info["start"] * bs
            if loss_info["n_conflict"] > 0:
                running_conflict += loss_info["conflict"] * loss_info["n_conflict"]

            evt_preds = torch.argmax(outputs["event_logits"], dim=1)
            event_correct += (evt_preds == batch_evt).sum().item()
            event_total += bs

            conf_mask = (batch_evt == 0) & (batch_conf >= 0)
            if conf_mask.any():
                conf_preds = torch.argmax(outputs["conflict_logits"][conf_mask], dim=1)
                conflict_correct += (conf_preds == batch_conf[conf_mask]).sum().item()
                conflict_total += int(conf_mask.sum().item())

            if batch_idx % GPU_LOG_INTERVAL == 0 or batch_idx == len(train_loader):
                stats = get_gpu_stats()
                backbone_lr = optimizer.param_groups[0]["lr"]
                head_lr = optimizer.param_groups[1]["lr"]
                print(
                    f"Epoch {epoch + 1:02d} batch {batch_idx:03d}/{len(train_loader):03d} | "
                    f"batch time: {batch_time:.2f}s | samples: {bs} | "
                    f"L_total: {loss_info['total']:.4f} | L_evt: {loss_info['event']:.4f} | "
                    f"L_conf: {loss_info['conflict']:.4f} (n={loss_info['n_conflict']}) | "
                    f"L_start: {loss_info['start']:.4f} | "
                    f"backbone_lr: {backbone_lr:.2e} | head_lr: {head_lr:.2e} | "
                    f"GPU alloc: {stats['allocated']} | peak alloc: {stats['max_allocated']}"
                )

        train_total_loss = running_total / max(event_total, 1)
        train_event_loss = running_event / max(event_total, 1)
        train_start_loss = running_start / max(event_total, 1)
        train_conflict_loss = running_conflict / max(conflict_total, 1) if conflict_total > 0 else float("nan")
        train_event_acc = event_correct / max(event_total, 1)
        train_conflict_acc = conflict_correct / max(conflict_total, 1) if conflict_total > 0 else float("nan")

        epoch_time = time.perf_counter() - epoch_start
        samples_per_sec = event_total / max(epoch_time, 1e-8)

        val_metrics = evaluate_loader(model, val_loader)
        val_event_acc = val_metrics["event_acc"]
        val_conflict_acc = val_metrics["conflict_acc"]
        val_start_mae = val_metrics["start_mae"]

        # Best-model selection score. Event is primary (0.7), conflict is a
        # secondary classification signal (0.3). Start-time MAE is not in
        # the score because it lives on a different scale.
        if not math.isnan(val_conflict_acc):
            val_score = 0.7 * val_event_acc + 0.3 * val_conflict_acc
        else:
            val_score = val_event_acc

        history.append({
            "epoch": epoch + 1,
            "train_total_loss": train_total_loss,
            "train_event_loss": train_event_loss,
            "train_conflict_loss": train_conflict_loss,
            "train_start_loss": train_start_loss,
            "train_event_acc": train_event_acc,
            "train_conflict_acc": train_conflict_acc,
            "val_event_acc": val_event_acc,
            "val_conflict_acc": val_conflict_acc,
            "val_start_mae": val_start_mae,
            "val_score": val_score,
        })
        print(
            f"Epoch {epoch + 1:02d} | Train L_total: {train_total_loss:.4f} | "
            f"Train Evt Acc: {train_event_acc:.4f} | Train Conf Acc: {train_conflict_acc:.4f} | "
            f"Val Evt Acc: {val_event_acc:.4f} | Val Conf Acc: {val_conflict_acc:.4f} | "
            f"Val Start MAE: {val_start_mae:.3f}s | Val Score: {val_score:.4f} | "
            f"Epoch Time: {epoch_time:.2f}s | Throughput: {samples_per_sec:.2f} samples/s"
        )
        print_gpu_stats(f"Epoch {epoch + 1:02d} GPU summary")

        if val_score > best_val_score:
            best_val_score = val_score
            patience_ctr = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "history": history,
                    "video_dim": model.video_encoder.out_dim,
                    "num_event_classes": NUM_EVENT_CLASSES,
                    "num_conflict_classes": NUM_CONFLICT_CLASSES,
                    "conflict_t_map": CONFLICT_T_MAP,
                },
                MODEL_PATH,
            )
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                print(f"Early stopping at epoch {epoch + 1}.")
                break

    print(f"\nBest joint multi-task fusion model saved to: {MODEL_PATH}")
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
        num_event_classes=NUM_EVENT_CLASSES,
        num_conflict_classes=NUM_CONFLICT_CLASSES,
    ).to(DEVICE)

    ckpt = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    metrics = evaluate_loader(model, test_loader)

    event_true = metrics["event_true"]
    event_pred = metrics["event_pred"]
    conflict_true = metrics["conflict_true"]
    conflict_pred = metrics["conflict_pred"]
    start_true = metrics["start_true"]
    start_pred = metrics["start_pred"]
    ids = metrics["ids"]

    print("\n" + "=" * 80)
    print("JOINT MULTI-TASK FUSION TEST RESULTS")

    # Event classification.
    print(f"\n[Event Type] Accuracy: {metrics['event_acc']:.4f}")
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

    # Conflict subtype: restricted to true-conflict rows with a valid label.
    conflict_mask = [(et == 0) and (ct >= 0) for et, ct in zip(event_true, conflict_true)]
    conflict_report = None
    if any(conflict_mask):
        ct_true = [c for c, m in zip(conflict_true, conflict_mask) if m]
        ct_pred = [c for c, m in zip(conflict_pred, conflict_mask) if m]
        present_labels = sorted(set(ct_true) | set(ct_pred))
        target_names = [CONFLICT_T_INV.get(i, f"idx_{i}") for i in present_labels]
        print(
            f"\n[Conflict Subtype] Accuracy on {len(ct_true)} conflict samples: "
            f"{metrics['conflict_acc']:.4f}"
        )
        conflict_report = classification_report(
            ct_true,
            ct_pred,
            labels=present_labels,
            target_names=target_names,
            output_dict=True,
            zero_division=0,
        )
        print(classification_report(
            ct_true,
            ct_pred,
            labels=present_labels,
            target_names=target_names,
            zero_division=0,
        ))
    else:
        print("\n[Conflict Subtype] No conflict samples with valid subtype labels in test set.")

    # Start time regression.
    print(f"\n[Start Time] MAE: {metrics['start_mae']:.3f}s | RMSE: {metrics['start_rmse']:.3f}s")

    # Predictions CSV. Conflict subtype is emitted only when the predicted
    # event is Conflict, matching the inference logic Matthew described.
    pred_conflict_labels = [
        CONFLICT_T_INV.get(cp) if ep == 0 else None
        for ep, cp in zip(event_pred, conflict_pred)
    ]
    true_conflict_labels = [
        CONFLICT_T_INV.get(ct) if et == 0 and ct >= 0 else None
        for et, ct in zip(event_true, conflict_true)
    ]

    out = pd.DataFrame(
        {
            "BDD_ID": ids,
            "event_true_idx": event_true,
            "event_pred_idx": event_pred,
            "event_true_label": [CLASS_MAP_STR[i] for i in event_true],
            "event_pred_label": [CLASS_MAP_STR[i] for i in event_pred],
            "conflict_true_label": true_conflict_labels,
            "conflict_pred_label": pred_conflict_labels,
            "start_true": start_true,
            "start_pred": start_pred,
            "start_abs_error": [abs(a - b) for a, b in zip(start_true, start_pred)],
        }
    )
    out.to_csv(PRED_PATH, index=False)

    with open(METRICS_PATH, "w") as f:
        json.dump(
            {
                "event": {
                    "accuracy": metrics["event_acc"],
                    "classification_report": event_report,
                },
                "conflict": {
                    "accuracy": metrics["conflict_acc"],
                    "classification_report": conflict_report,
                },
                "start_time": {
                    "mae_seconds": metrics["start_mae"],
                    "rmse_seconds": metrics["start_rmse"],
                },
            },
            f,
            indent=2,
        )

    print(f"\nSaved predictions to: {PRED_PATH}")
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
