
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

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Normalize, Resize
from tqdm import tqdm

from pytorchvideo.data.encoded_video import EncodedVideo
from pytorchvideo.models.hub import slowfast_r50
from pytorchvideo.transforms import UniformTemporalSubsample

# =========================================================
# CONFIG
# =========================================================
LABELS_DIR = Path("data/100k/train")
BDD_META_PATH = Path("data/downloaded_videos_meta.csv")
VIDEO_DIR = Path("data/annotated_videos_only")

CACHE_DIR = Path("data/slowfast_video_only")
TENSOR_DIR = CACHE_DIR / "transformed_tensors"
EMBED_DIR = CACHE_DIR / "embeddings"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
TENSOR_DIR.mkdir(parents=True, exist_ok=True)
EMBED_DIR.mkdir(parents=True, exist_ok=True)

SPLITS_PATH = CACHE_DIR / "video_only_splits.csv"
MODEL_PATH = CACHE_DIR / "slowfast_video_head_best.pth"

CLIP_START_SEC = 10
CLIP_END_SEC = 18
RESIZE_HW = (256, 455)
BATCH_SIZE = 16
EPOCHS = 15
LEARNING_RATE = 1e-3
SEED = 1
NUM_CLASSES = 4

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
class SlowFastTransform:
    def __init__(self) -> None:
        self.force_landscape = Resize(RESIZE_HW)
        self.normalize = Normalize([0.45] * 3, [0.225] * 3)
        self.slow_subsample = UniformTemporalSubsample(8)
        self.fast_subsample = UniformTemporalSubsample(32)

    def __call__(self, x: torch.Tensor) -> List[torch.Tensor]:
        x = x / 255.0
        x = self.force_landscape(x)
        x = x.permute(1, 0, 2, 3)
        x = self.normalize(x)
        x = x.permute(1, 0, 2, 3)
        return [self.slow_subsample(x), self.fast_subsample(x)]


class SlowFastEmbeddingExtractor(nn.Module):
    """
    Frozen SlowFast backbone that outputs the 2304-d pooled video embedding.
    """
    def __init__(self) -> None:
        super().__init__()
        self.backbone = slowfast_r50(pretrained=True)
        self.backbone.blocks[6].proj = nn.Identity()

    def forward(self, video_pathway: List[torch.Tensor]) -> torch.Tensor:
        return self.backbone(video_pathway)


class VideoOnlySlowFastHead(nn.Module):
    """
    Small MLP head on top of precomputed SlowFast embeddings.
    This isolates the video branch first.
    """
    def __init__(self, in_dim: int = 2304, num_classes: int = NUM_CLASSES) -> None:
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, video_embedding: torch.Tensor) -> torch.Tensor:
        return self.head(video_embedding)


class FutureFusionHead(nn.Module):
    """
    Placeholder for later fusion with kinematics/context.
    Keep this for the next phase after video-only is stable.
    """
    def __init__(self, video_dim: int, aux_dim: int, num_classes: int = NUM_CLASSES) -> None:
        super().__init__()
        self.video_proj = nn.Sequential(
            nn.Linear(video_dim, 256),
            nn.ReLU(),
        )
        self.aux_proj = nn.Sequential(
            nn.Linear(aux_dim, 128),
            nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(256 + 128, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, video_embedding: torch.Tensor, aux_features: torch.Tensor) -> torch.Tensor:
        v = self.video_proj(video_embedding)
        a = self.aux_proj(aux_features)
        return self.classifier(torch.cat([v, a], dim=1))


# =========================================================
# DATASETS
# =========================================================
class VideoEmbeddingDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame) -> None:
        self.df = dataframe.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        embedding = torch.load(row["embedding_path"], map_location="cpu", weights_only=False).float()
        target = int(row["target_idx"])
        bdd_id = row["BDD_ID"]
        return embedding, target, bdd_id


# =========================================================
# METADATA / SPLITS
# =========================================================
def build_video_metadata() -> pd.DataFrame:
    bdd_df = pd.read_csv(BDD_META_PATH).copy()
    bdd_df["BDD_ID"] = bdd_df["BDD_ID"].astype(str)

    # Keep only labels that actually have videos available.
    video_map = {p.stem: p for p in VIDEO_DIR.iterdir() if p.suffix == ".mov"}
    df = bdd_df[bdd_df["BDD_ID"].isin(video_map.keys())].copy()

    # Expect EVENT_TYPE in {1,2,3,4}; convert to 0-based for PyTorch.
    df["target_idx"] = df["EVENT_TYPE"].astype(int) - 1
    df["video_path"] = df["BDD_ID"].map(lambda x: str(video_map[x]))
    return df.reset_index(drop=True)


def build_group_splits(df: pd.DataFrame, seed: int = SEED) -> pd.DataFrame:
    groups = df["BDD_ID"].astype(str).to_numpy()
    y = df["target_idx"].to_numpy()

    gss1 = GroupShuffleSplit(n_splits=1, train_size=0.70, random_state=seed)
    train_idx, temp_idx = next(gss1.split(df, y, groups=groups))

    temp_df = df.iloc[temp_idx].copy()
    temp_groups = temp_df["BDD_ID"].astype(str).to_numpy()
    temp_y = temp_df["target_idx"].to_numpy()

    gss2 = GroupShuffleSplit(n_splits=1, train_size=0.50, random_state=seed)
    val_rel, test_rel = next(gss2.split(temp_df, temp_y, groups=temp_groups))

    df = df.copy()
    df["split"] = "train"
    df.iloc[temp_idx, df.columns.get_loc("split")] = "temp"
    df.iloc[temp_idx[val_rel], df.columns.get_loc("split")] = "val"
    df.iloc[temp_idx[test_rel], df.columns.get_loc("split")] = "test"

    return df


# =========================================================
# PRECOMPUTE VIDEO ARTIFACTS
# =========================================================
def load_and_transform_video(video_path: Path, transform: SlowFastTransform) -> Optional[List[torch.Tensor]]:
    try:
        video = EncodedVideo.from_path(str(video_path))
        clip = video.get_clip(start_sec=CLIP_START_SEC, end_sec=CLIP_END_SEC)
        if clip is None or "video" not in clip or clip["video"] is None:
            print(f"Skipping {video_path.name}: empty clip")
            return None
        frames = transform(clip["video"])
        return [f.cpu() for f in frames]
    except Exception as e:
        print(f"Failed on video {video_path.stem}: {e}")
        return None


@torch.no_grad()
def precompute_tensors_and_embeddings(df: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "=" * 60)
    print("PRECOMPUTING SLOWFAST TENSORS AND VIDEO EMBEDDINGS")

    transform = SlowFastTransform()
    extractor = SlowFastEmbeddingExtractor().to(DEVICE)
    extractor.eval()
    for p in extractor.parameters():
        p.requires_grad = False

    df = df.copy()
    df["tensor_path"] = None
    df["embedding_path"] = None

    kept_rows = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Precomputing video artifacts"):
        bdd_id = row["BDD_ID"]
        video_path = Path(row["video_path"])
        tensor_path = TENSOR_DIR / f"{bdd_id}.pt"
        embedding_path = EMBED_DIR / f"{bdd_id}.pt"

        if not tensor_path.exists():
            frames = load_and_transform_video(video_path, transform)
            if frames is None:
                continue
            torch.save({"slow": frames[0], "fast": frames[1]}, tensor_path)
        else:
            saved = torch.load(tensor_path, map_location="cpu", weights_only=False)
            frames = [saved["slow"], saved["fast"]]

        if not embedding_path.exists():
            frames_device = [f.unsqueeze(0).to(DEVICE) for f in frames]
            embedding = extractor(frames_device).squeeze(0).detach().cpu()
            torch.save(embedding, embedding_path)

            del embedding, frames_device
            if DEVICE.type == "cuda":
                torch.cuda.empty_cache()
            elif DEVICE.type == "mps":
                torch.mps.empty_cache()

        df.at[idx, "tensor_path"] = str(tensor_path)
        df.at[idx, "embedding_path"] = str(embedding_path)
        kept_rows.append(idx)

    df = df.loc[kept_rows].reset_index(drop=True)
    df.to_csv(SPLITS_PATH, index=False)
    print(f"Saved video-only metadata to: {SPLITS_PATH}")
    return df


# =========================================================
# TRAINING / EVALUATION
# =========================================================
def make_loaders(df: pd.DataFrame) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_df = df[df["split"] == "train"].copy()
    val_df = df[df["split"] == "val"].copy()
    test_df = df[df["split"] == "test"].copy()

    train_loader = DataLoader(VideoEmbeddingDataset(train_df), batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=False)
    val_loader = DataLoader(VideoEmbeddingDataset(val_df), batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=False)
    test_loader = DataLoader(VideoEmbeddingDataset(test_df), batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=False)
    return train_loader, val_loader, test_loader


def evaluate_loader(model: nn.Module, loader: DataLoader) -> Tuple[float, List[int], List[int], List[str]]:
    model.eval()
    y_true: List[int] = []
    y_pred: List[int] = []
    ids: List[str] = []

    with torch.no_grad():
        for batch_emb, batch_targets, batch_ids in loader:
            batch_emb = batch_emb.to(DEVICE)
            logits = model(batch_emb)
            preds = torch.argmax(logits, dim=1).cpu().numpy()

            y_true.extend(batch_targets.numpy().tolist())
            y_pred.extend(preds.tolist())
            ids.extend(list(batch_ids))

    acc = accuracy_score(y_true, y_pred)
    return acc, y_true, y_pred, ids


def train_video_only_model(df: pd.DataFrame) -> VideoOnlySlowFastHead:
    train_loader, val_loader, _ = make_loaders(df)

    model = VideoOnlySlowFastHead().to(DEVICE)

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

        for batch_emb, batch_targets, _ in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}", leave=False):
            batch_emb = batch_emb.to(DEVICE)
            batch_targets = batch_targets.to(DEVICE)

            optimizer.zero_grad()
            logits = model(batch_emb)
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
            torch.save(model.state_dict(), MODEL_PATH)

    print(f"\nBest video-only model saved to: {MODEL_PATH}")
    return model


def evaluate_test(model: VideoOnlySlowFastHead, df: pd.DataFrame) -> None:
    _, _, test_loader = make_loaders(df)
    acc, y_true, y_pred, ids = evaluate_loader(model, test_loader)

    print("\n" + "=" * 80)
    print("VIDEO-ONLY SLOWFAST TEST RESULTS")
    print(f"Test Accuracy: {acc:.4f}")
    print(classification_report(y_true, y_pred, target_names=[CLASS_MAP_STR[i] for i in range(NUM_CLASSES)]))

    results = pd.DataFrame({
        "BDD_ID": ids,
        "true_idx": y_true,
        "pred_idx": y_pred,
        "true_label": [CLASS_MAP_STR[i] for i in y_true],
        "pred_label": [CLASS_MAP_STR[i] for i in y_pred],
    })
    results_path = CACHE_DIR / "video_only_test_predictions.csv"
    results.to_csv(results_path, index=False)
    print(f"Saved per-video test predictions to: {results_path}")
    print("=" * 80)


def main() -> None:
    set_seed(SEED)

    df = build_video_metadata()
    df = build_group_splits(df, seed=SEED)
    df = precompute_tensors_and_embeddings(df)

    model = train_video_only_model(df)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False))
    evaluate_test(model, df)

    print("\nNext step:")
    print("- keep these saved SlowFast embeddings")
    print("- train your kinematics/context model separately")
    print("- concatenate video embedding + kinematics embedding with a small fusion head")


if __name__ == "__main__":
    main()
