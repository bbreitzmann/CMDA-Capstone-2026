"""Smoke test for the multi-task fusion model.

Runs the model surgery end-to-end on fake tensors: no videos, no data files
required. Confirms:
  - all three heads produce the expected output shapes
  - the masked conflict loss handles batches with zero conflict samples
  - the masked conflict loss handles batches with all conflict samples
  - a full forward + backward + optimizer step runs without error

If this script exits 0, the multi-task changes are structurally correct and
the only thing standing between you and training is having the real data.
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn

# Make the training script importable as a module.
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

# Importing the training script triggers its top-level prints. That's fine
# for a smoke test. If you want quiet, wrap this in contextlib.redirect_stdout.
from mmaction2_slowfast_joint_multitask_fusion import (  # noqa: E402
    CLIP_LEN,
    CROP_SIZE,
    FocalLoss,
    JointEndToEndFusionModel,
    KinematicsBranch,
    NUM_CONFLICT_CLASSES,
    NUM_EVENT_CLASSES,
    compute_multitask_loss,
)


class FakeVideoEncoder(nn.Module):
    """Stand-in for MMAction2SlowFastFeatureExtractor.

    Returns a random tensor of a configurable width so the rest of the model
    gets wired up exactly the same way it would with the real SlowFast
    backbone, just without needing the checkpoint or any video tensors.
    """

    def __init__(self, out_dim: int = 2304) -> None:
        super().__init__()
        self.out_dim = out_dim
        # One real parameter so optimizer has something to step on.
        self.dummy = nn.Linear(16, out_dim)

    def forward(self, video_inputs: torch.Tensor) -> torch.Tensor:
        # video_inputs is [B, N, C, T, H, W]; we only need B.
        b = video_inputs.shape[0]
        seed = video_inputs.mean(dim=(1, 2, 3, 4, 5), keepdim=False).unsqueeze(1)
        seed = seed.expand(b, 16)
        return self.dummy(seed)


def build_model(device: torch.device) -> JointEndToEndFusionModel:
    video_encoder = FakeVideoEncoder(out_dim=2304)
    model = JointEndToEndFusionModel(
        video_encoder=video_encoder,
        in_chans_ts=7,   # matches your X_ts shape (2500, 7, 256)
        in_chans_ctx=164,
        num_event_classes=NUM_EVENT_CLASSES,
        num_conflict_classes=NUM_CONFLICT_CLASSES,
    ).to(device)
    return model


def make_fake_batch(batch_size: int, event_targets: list, conflict_targets: list, device: torch.device):
    video = torch.randn(batch_size, 1, 3, CLIP_LEN, CROP_SIZE, CROP_SIZE, device=device)
    x_ts = torch.randn(batch_size, 7, 256, device=device)
    x_ctx = torch.randn(batch_size, 164, device=device)
    evt = torch.tensor(event_targets, dtype=torch.long, device=device)
    conf = torch.tensor(conflict_targets, dtype=torch.long, device=device)
    start = torch.rand(batch_size, device=device) * 40.0  # start times 0-40s
    return video, x_ts, x_ctx, evt, conf, start


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[smoke] device: {device}")

    model = build_model(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[smoke] trainable params: {n_params:,}")

    event_criterion = FocalLoss(gamma=2.0)
    conflict_criterion = nn.CrossEntropyLoss()
    start_criterion = nn.SmoothL1Loss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # --- Case 1: mixed batch with some conflicts and some non-conflicts ---
    print("\n[smoke] case 1: mixed batch (2 conflicts, 2 non-conflicts)")
    video, x_ts, x_ctx, evt, conf, start = make_fake_batch(
        batch_size=4,
        event_targets=[0, 1, 0, 3],       # 2 conflicts (idx 0), 1 bump, 1 not-SCE
        conflict_targets=[5, -1, 12, -1], # subtypes for conflicts, -1 elsewhere
        device=device,
    )
    optimizer.zero_grad()
    outputs = model(video, x_ts, x_ctx)
    assert outputs["event_logits"].shape == (4, NUM_EVENT_CLASSES), outputs["event_logits"].shape
    assert outputs["conflict_logits"].shape == (4, NUM_CONFLICT_CLASSES), outputs["conflict_logits"].shape
    assert outputs["start_pred"].shape == (4,), outputs["start_pred"].shape
    print(f"[smoke]   event_logits: {tuple(outputs['event_logits'].shape)}")
    print(f"[smoke]   conflict_logits: {tuple(outputs['conflict_logits'].shape)}")
    print(f"[smoke]   start_pred: {tuple(outputs['start_pred'].shape)}")

    loss, info = compute_multitask_loss(
        outputs, evt, conf, start, event_criterion, conflict_criterion, start_criterion,
    )
    assert info["n_conflict"] == 2, info
    loss.backward()
    optimizer.step()
    print(f"[smoke]   loss: total={info['total']:.4f} evt={info['event']:.4f} "
          f"conf={info['conflict']:.4f} start={info['start']:.4f} n_conf={info['n_conflict']}")

    # --- Case 2: batch with zero conflict samples ---
    print("\n[smoke] case 2: zero-conflict batch (conflict loss must not blow up)")
    video, x_ts, x_ctx, evt, conf, start = make_fake_batch(
        batch_size=3,
        event_targets=[1, 2, 3],        # bump, hard brake, not-SCE
        conflict_targets=[-1, -1, -1],
        device=device,
    )
    optimizer.zero_grad()
    outputs = model(video, x_ts, x_ctx)
    loss, info = compute_multitask_loss(
        outputs, evt, conf, start, event_criterion, conflict_criterion, start_criterion,
    )
    assert info["n_conflict"] == 0, info
    assert info["conflict"] == 0.0, f"conflict loss should be 0.0 when no conflicts, got {info['conflict']}"
    loss.backward()
    optimizer.step()
    print(f"[smoke]   loss: total={info['total']:.4f} evt={info['event']:.4f} "
          f"conf={info['conflict']:.4f} start={info['start']:.4f} n_conf={info['n_conflict']}")

    # --- Case 3: batch with all conflict samples ---
    print("\n[smoke] case 3: all-conflict batch")
    video, x_ts, x_ctx, evt, conf, start = make_fake_batch(
        batch_size=3,
        event_targets=[0, 0, 0],
        conflict_targets=[2, 10, 16],   # E, Q, Y
        device=device,
    )
    optimizer.zero_grad()
    outputs = model(video, x_ts, x_ctx)
    loss, info = compute_multitask_loss(
        outputs, evt, conf, start, event_criterion, conflict_criterion, start_criterion,
    )
    assert info["n_conflict"] == 3, info
    loss.backward()
    optimizer.step()
    print(f"[smoke]   loss: total={info['total']:.4f} evt={info['event']:.4f} "
          f"conf={info['conflict']:.4f} start={info['start']:.4f} n_conf={info['n_conflict']}")

    print("\n[smoke] all cases passed. multi-task model surgery is structurally sound.")
    print("[smoke] remaining blocker is the videos + bdd_sce.csv from Matthew.")


if __name__ == "__main__":
    main()
