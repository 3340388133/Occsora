#!/usr/bin/env python3
"""Evaluate OccSora architecture models (baseline / +STCA / +SADS / full).

Metrics requested:
- Inference time, speed (samples/sec)
- Diversity
- Temporal consistency
- Physical consistency (occupancy-based)
- mIoU (vs reference token decoded via VQVAE)
- FID (2D feature FID on latents)
- FVD (3D feature FID on latents; "FVD-like")
- CD (Chamfer distance on decoded occupancy point clouds)

Notes
- True FVD usually uses a pretrained video network on RGB videos (e.g. I3D).
  Here we compute an FVD-like score by extracting 3D features from latent
  tensors and applying the standard Fréchet distance.
- mIoU/CD/physics are computed against a *reference token* (from VQVAE token
    directory) that corresponds to each chosen condition index.

Physics consistency definition (evaluation-only):
- Ground support score: fraction of occupied voxels at z>0 that are supported
    by an occupied voxel directly below (z-1). Higher is better.
- Motion smoothness score: 1/(1+mean(|acceleration|)) on per-voxel binary
    occupancy across frames. Higher is better.
The final physics score is the average of these two.

Example:
    python3 evaluate_all_metrics.py --num-samples 4 --num-steps 50 --cond-indices 0,1,2,3
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import linalg
from scipy.spatial import cKDTree

# Local imports (repo)
import sys
sys.path.insert(0, "/root/OccSora-main")

from models_stca import DiT_STCA_models
from diffusion import create_diffusion
from Project.innovations.sads import SADSSampler
from Project.consistency.metrics import TemporalConsistencyMetric

from mmengine import Config
from mmengine.registry import MODELS
import model as model_module  # noqa: F401 (register mmengine modules)


# -------------------------
# Feature extractors + FID
# -------------------------


class FeatureExtractor3D(nn.Module):
    """Simple 3D CNN over latents (B, C, D, H, W)."""

    def __init__(self, in_channels: int = 128):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, 64, 3, stride=2, padding=1)
        self.conv2 = nn.Conv3d(64, 128, 3, stride=2, padding=1)
        self.conv3 = nn.Conv3d(128, 256, 3, stride=1, padding=1)
        self.pool = nn.AdaptiveAvgPool3d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        return x.view(x.size(0), -1)


class FeatureExtractor2D(nn.Module):
    """2D CNN over latents averaged over depth/time: (B, C, H, W)."""

    def __init__(self, in_channels: int = 128):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(128, 256, 3, stride=1, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        return x.view(x.size(0), -1)


def frechet_distance(real_features: np.ndarray, fake_features: np.ndarray) -> float:
    """FID-style Fréchet distance between two feature distributions."""

    real_features = np.asarray(real_features, dtype=np.float64)
    fake_features = np.asarray(fake_features, dtype=np.float64)

    mu_real = np.mean(real_features, axis=0)
    mu_fake = np.mean(fake_features, axis=0)

    def _cov(feats: np.ndarray) -> np.ndarray:
        if feats.shape[0] < 2:
            d = int(feats.shape[1])
            return np.eye(d, dtype=np.float64) * 1e-6
        return np.cov(feats, rowvar=False)

    sigma_real = _cov(real_features)
    sigma_fake = _cov(fake_features)

    sigma_real = np.atleast_2d(sigma_real)
    sigma_fake = np.atleast_2d(sigma_fake)

    diff = mu_real - mu_fake
    covmean, _ = linalg.sqrtm(sigma_real @ sigma_fake, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    return float(diff @ diff + np.trace(sigma_real + sigma_fake - 2 * covmean))


# -------------------------
# Sampling
# -------------------------


@dataclass
class ModelSpec:
    key: str
    name: str
    ckpt: str
    use_stca: bool
    use_sads: bool


def _load_ckpt_into_model(model: torch.nn.Module, ckpt_path: str, device: torch.device) -> None:
    if not ckpt_path or not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    if isinstance(ckpt, dict):
        if "model_state_dict" in ckpt:
            state = ckpt["model_state_dict"]
        elif "model" in ckpt:
            state = ckpt["model"]
        elif "ema" in ckpt:
            state = ckpt["ema"]
        else:
            state = ckpt
    else:
        state = ckpt
    model.load_state_dict(state, strict=False)


def sample_one(
    model: torch.nn.Module,
    device: torch.device,
    condition_vec: np.ndarray,
    seed: int,
    num_steps: int,
    cfg_scale: float,
    use_sads: bool,
) -> Tuple[np.ndarray, float, Optional[int]]:
    """Return (sample_latent, elapsed_seconds, actual_steps)."""

    torch.manual_seed(seed)

    # CFG batch convention used in `sample_architecture.py`
    y = np.stack([condition_vec, condition_vec], axis=0).astype(np.float32)
    y = torch.tensor(y, device=device)

    z = torch.randn((2, 128, 4, 25, 25), device=device)
    model_kwargs = dict(y=y, cfg_scale=cfg_scale)

    torch.cuda.synchronize() if device.type == "cuda" else None
    start = time.time()

    if use_sads:
        sampler = SADSSampler(
            num_timesteps=1000,
            num_inference_steps=num_steps,
            device=device,
        ).to(device)
        with torch.no_grad():
            out = sampler.sample(
                model=model.forward_with_cfg,
                shape=z.shape,
                noise=z,
                model_kwargs=model_kwargs,
                progress=False,
            )
        actual_steps = None
    else:
        diffusion = create_diffusion(str(num_steps))
        with torch.no_grad():
            out = diffusion.p_sample_loop(
                model.forward_with_cfg,
                z.shape,
                z,
                clip_denoised=False,
                model_kwargs=model_kwargs,
                progress=False,
                device=device,
            )
        actual_steps = num_steps

    torch.cuda.synchronize() if device.type == "cuda" else None
    elapsed = time.time() - start

    out, _ = out.chunk(2, dim=0)
    sample = out.detach().cpu().numpy()  # (1,128,4,25,25)

    return sample, float(elapsed), actual_steps


# -------------------------
# Latent-level metrics
# -------------------------


def diversity_rmse(samples: np.ndarray, max_pairs: int = 100) -> float:
    samples = np.asarray(samples)
    if samples.ndim == 5:
        n = samples.shape[0]
    elif samples.ndim == 4:
        samples = samples[None, ...]
        n = 1
    else:
        return 0.0

    if n < 2:
        return 0.0

    flat = samples.reshape(n, -1).astype(np.float64)
    total_pairs = n * (n - 1) // 2
    n_pairs = min(max_pairs, total_pairs)

    dists: List[float] = []
    for _ in range(n_pairs):
        i, j = np.random.choice(n, 2, replace=False)
        d = np.sqrt(np.mean((flat[i] - flat[j]) ** 2))
        dists.append(float(d))

    return float(np.mean(dists)) if dists else 0.0


def temporal_consistency_latent(samples: np.ndarray, device: torch.device) -> Dict[str, float]:
    """Compute temporal MAD on latents (treat depth=4 as time)."""

    x = torch.from_numpy(np.asarray(samples)).float()
    if x.ndim == 4:
        x = x.unsqueeze(0)
    # (B,C,T,H,W) -> (B,T,C,H,W)
    x = x.permute(0, 2, 1, 3, 4).to(device)

    metric = TemporalConsistencyMetric(device=str(device))
    res = metric.compute(x)
    mad = float(res.details.get("mean_absolute_difference", res.value))
    score = float(1.0 / (1.0 + mad))

    out = {
        "temporal_mad": mad,
        "temporal_score": score,
        "flicker_index": float(res.details.get("flicker_index", 0.0)),
        "motion_smoothness": float(res.details.get("motion_smoothness", 0.0)),
    }
    return out


def mean_std(values: List[float]) -> Tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return float("nan"), float("nan")
    return float(v.mean()), float(v.std(ddof=0))


# -------------------------
# VQVAE decoding (chunked)
# -------------------------


@torch.no_grad()
def load_vqvae(vqvae_ckpt: str, vqvae_config: str, device: torch.device):
    cfg = Config.fromfile(vqvae_config)
    vq = MODELS.build(cfg.model)

    ckpt = torch.load(vqvae_ckpt, map_location="cpu")
    state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    vq.load_state_dict(state_dict, strict=False)
    vq = vq.to(device).eval()
    return vq


@torch.no_grad()
def decode_latent_to_semantic(
    vq,
    latent_128: torch.Tensor,
    chunk_hw: int = 5000,
) -> torch.Tensor:
    """Decode a single sample latent to semantic occupancy labels.

    Args:
        latent_128: (1,128,4,25,25) or (B,128,4,25,25)

    Returns:
        labels: (B, F, H, W, D) int16, where F=32, H=W=200, D=16
    """

    if latent_128.ndim != 5:
        raise ValueError(f"Expected 5D latent, got {tuple(latent_128.shape)}")

    # Match existing decode logic: only the first 64 channels are used as z.
    z = latent_128[:, :64]

    x = vq.decoder_gpt(vq.post_vq_conv(z))  # (B, C=128, F=32, H=200, W=200)

    B, C, F_out, H, W = x.shape
    D = 16
    expansion = int(getattr(vq, "expansion", 8))
    if C != D * expansion:
        raise RuntimeError(f"Unexpected decoder channel count: C={C}, expected {D}*{expansion}")

    template = vq.class_embeds.weight.T  # (expansion, num_cls)
    num_cls = template.shape[1]

    # We'll decode frame-by-frame, chunked over H*W.
    labels = torch.empty((B, F_out, H, W, D), dtype=torch.int16, device="cpu")

    x = x.permute(0, 2, 3, 4, 1).contiguous()  # (B,F,H,W,C)

    for b in range(B):
        for f in range(F_out):
            xf = x[b, f]  # (H,W,C)
            xf = xf.view(H * W, D, expansion)  # (HW, D, exp)

            pred_f = torch.empty((H * W, D), dtype=torch.int16, device="cpu")

            for start in range(0, H * W, chunk_hw):
                end = min(start + chunk_hw, H * W)
                chunk = xf[start:end].to(template.device)  # (chunk, D, exp)

                # (chunk, D, exp) @ (exp, num_cls) -> (chunk, D, num_cls)
                sim = torch.matmul(chunk, template)  # float
                pred = torch.argmax(sim, dim=-1).to(torch.int16).cpu()  # (chunk, D)
                pred_f[start:end] = pred

            labels[b, f] = pred_f.view(H, W, D)

    return labels


# -------------------------
# Occupancy-level metrics
# -------------------------


def compute_miou(pred: np.ndarray, gt: np.ndarray, num_classes: int, ignore: Optional[int] = None) -> float:
    pred = pred.reshape(-1)
    gt = gt.reshape(-1)

    if ignore is not None:
        m = gt != ignore
        pred = pred[m]
        gt = gt[m]

    ious: List[float] = []
    for c in range(0, num_classes):
        if ignore is not None and c == ignore:
            continue
        pred_c = pred == c
        gt_c = gt == c
        inter = np.logical_and(pred_c, gt_c).sum()
        union = np.logical_or(pred_c, gt_c).sum()
        if union == 0:
            continue
        ious.append(float(inter) / float(union))

    return float(np.mean(ious)) if ious else 0.0


def occupancy_points(
    labels: np.ndarray,
    empty_label: int,
    max_points: int = 20000,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Convert (H,W,D) labels to sampled occupied point coordinates (N,3)."""

    mask = labels != empty_label
    pts = np.argwhere(mask)  # (N,3) with order (H,W,D)
    if pts.size == 0:
        return pts.astype(np.float32)

    if max_points and pts.shape[0] > max_points:
        rng = rng or np.random.default_rng(0)
        idx = rng.choice(pts.shape[0], size=max_points, replace=False)
        pts = pts[idx]

    return pts.astype(np.float32)


def chamfer_distance(a: np.ndarray, b: np.ndarray, empty_penalty: float = 1000.0) -> float:
    if a.size == 0 and b.size == 0:
        return 0.0
    if a.size == 0 or b.size == 0:
        return float(empty_penalty)

    ta = cKDTree(a)
    tb = cKDTree(b)
    da, _ = tb.query(a, k=1)
    db, _ = ta.query(b, k=1)
    return float(np.mean(da) + np.mean(db))


def occupancy_metrics(
    pred_occ: np.ndarray,
    ref_occ: np.ndarray,
    empty_label: int,
    num_classes: int,
    cd_frame_stride: int = 4,
    cd_max_points: int = 15000,
) -> Dict[str, float]:
    """Compute mIoU, CD on decoded occupancy sequences.

    Args:
        pred_occ/ref_occ: (F,H,W,D)
    """

    miou = compute_miou(pred_occ, ref_occ, num_classes=num_classes, ignore=empty_label)

    # Chamfer distance over a subset of frames
    rng = np.random.default_rng(0)
    cds: List[float] = []
    for f in range(0, pred_occ.shape[0], max(1, cd_frame_stride)):
        a = occupancy_points(pred_occ[f], empty_label=empty_label, max_points=cd_max_points, rng=rng)
        b = occupancy_points(ref_occ[f], empty_label=empty_label, max_points=cd_max_points, rng=rng)
        cds.append(chamfer_distance(a, b))

    cd = float(np.mean(cds)) if cds else float("inf")

    return {
        "miou": float(miou),
        "cd": cd,
    }


def physics_consistency_occupancy(
    decoded: np.ndarray,
    empty_label: int,
    ref_occ: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Occupancy-based physics consistency on decoded semantic labels.

    Args:
        decoded: (F,H,W,D) int labels
        ref_occ: optional reference occupancy labels (F,H,W,D) for non-triviality / density matching
    """

    # Binary occupancy: treat everything except empty_label as occupied
    occ = decoded != empty_label
    occ_ratio = float(np.mean(occ))

    # Guard against the trivial "almost empty" solution
    if occ_ratio < 1e-8:
        return {
            "physics_score": 0.0,
            "occ_ratio": occ_ratio,
            "occupancy_match_score": 0.0,
            "ground_support_score": 0.0,
            "unsupported_ratio": 1.0,
            "motion_smooth_score": 0.0,
            "motion_accel": 0.0,
        }

    occupancy_match_score = 1.0
    ref_ratio = None
    if ref_occ is not None:
        ref_mask = ref_occ != empty_label
        ref_ratio = float(np.mean(ref_mask))
        # Relative density error; clamp into [0,1] score
        rel_err = abs(occ_ratio - ref_ratio) / max(ref_ratio, 1e-8)
        occupancy_match_score = float(max(0.0, 1.0 - rel_err))

    # Ground support: occupied at z>0 should have support at z-1
    above = occ[..., 1:]
    below = occ[..., :-1]
    unsupported = above & (~below)
    denom = int(above.sum())
    if denom <= 0:
        # No occupied voxels above ground -> treat as neutral (not a violation)
        denom = 1
    unsupported_ratio = float(unsupported.sum()) / float(denom)
    ground_support_score = float(1.0 - unsupported_ratio)

    # Motion smoothness on occupancy changes (acceleration)
    occ_f = occ.astype(np.float32)
    if occ_f.shape[0] >= 3:
        v = occ_f[1:] - occ_f[:-1]
        a = v[1:] - v[:-1]
        accel = float(np.mean(np.abs(a)))
    else:
        accel = 0.0
    motion_smooth_score = float(1.0 / (1.0 + accel))

    # Weighted: encourage non-trivial occupancy + physical plausibility
    physics_score = float(
        0.5 * occupancy_match_score + 0.25 * ground_support_score + 0.25 * motion_smooth_score
    )

    return {
        "physics_score": physics_score,
        "occ_ratio": occ_ratio,
        "occupancy_match_score": occupancy_match_score,
        "ground_support_score": ground_support_score,
        "unsupported_ratio": unsupported_ratio,
        "motion_smooth_score": motion_smooth_score,
        "motion_accel": float(accel),
    }


# -------------------------
# Utilities
# -------------------------


def pick_ckpt(*candidates: str) -> str:
    for p in candidates:
        if p and os.path.exists(p):
            return p
    # Try glob newest if directory provided
    for p in candidates:
        if p and os.path.isdir(p):
            pts = sorted(Path(p).glob("*.pt"), key=lambda x: x.stat().st_mtime, reverse=True)
            if pts:
                return str(pts[0])
    return ""


def load_condition_vec(cond_idx: int, gt_mode_dir: str) -> np.ndarray:
    p = os.path.join(gt_mode_dir, f"i_iter_{cond_idx}.npy")
    if not os.path.exists(p):
        raise FileNotFoundError(f"Condition file not found: {p}")
    v = np.load(p).astype(np.float32)
    # Pad/truncate to 64
    if v.shape[0] < 64:
        rep = (64 // v.shape[0]) + 1
        v = np.tile(v, rep)[:64]
    else:
        v = v[:64]
    return v


def load_real_latents(token_dir: str, num_samples: int) -> torch.Tensor:
    files = sorted([f for f in os.listdir(token_dir) if f.endswith('.npy')])
    if not files:
        raise FileNotFoundError(f"No token npy files under: {token_dir}")

    arrs: List[np.ndarray] = []
    for i in range(num_samples):
        token = np.load(os.path.join(token_dir, files[i % len(files)])).astype(np.float32)  # (64,4,25,25)
        token = np.concatenate([token, token], axis=0)  # (128,4,25,25)
        arrs.append(token)

    x = torch.from_numpy(np.stack(arrs, axis=0)).float()  # (N,128,4,25,25)
    return x


def ensure_dir(p: str) -> None:
    Path(p).mkdir(parents=True, exist_ok=True)


def _fmt_mean_std(mean: float, std: float, fmt: str) -> str:
    if np.isnan(mean) or np.isnan(std):
        return "nan"
    return f"{mean:{fmt}}±{std:{fmt}}"


def write_tables(summary: Dict[str, Dict[str, Dict[str, float]]], out_dir: str) -> None:
    md_path = os.path.join(out_dir, "metrics_table.md")
    tex_path = os.path.join(out_dir, "metrics_table.tex")

    headers = [
        "Method",
        "Time(s)↓",
        "Speed(samples/s)↑",
        "Diversity↑",
        "TempScore↑",
        "PhysScore↑",
        "mIoU↑",
        "FID↓",
        "FVD↓",
        "CD↓",
    ]

    rows = []
    for key in ["baseline", "stca", "sads", "full"]:
        m = summary[key]
        rows.append([
            key,
            _fmt_mean_std(m.get("avg_time", {}).get("mean", float("nan")), m.get("avg_time", {}).get("std", float("nan")), ".3f"),
            _fmt_mean_std(m.get("throughput", {}).get("mean", float("nan")), m.get("throughput", {}).get("std", float("nan")), ".3f"),
            _fmt_mean_std(m.get("diversity", {}).get("mean", float("nan")), m.get("diversity", {}).get("std", float("nan")), ".4f"),
            _fmt_mean_std(m.get("temporal_score", {}).get("mean", float("nan")), m.get("temporal_score", {}).get("std", float("nan")), ".4f"),
            _fmt_mean_std(m.get("physics_score", {}).get("mean", float("nan")), m.get("physics_score", {}).get("std", float("nan")), ".4f"),
            _fmt_mean_std(m.get("miou", {}).get("mean", float("nan")), m.get("miou", {}).get("std", float("nan")), ".4f"),
            _fmt_mean_std(m.get("fid", {}).get("mean", float("nan")), m.get("fid", {}).get("std", float("nan")), ".2f"),
            _fmt_mean_std(m.get("fvd", {}).get("mean", float("nan")), m.get("fvd", {}).get("std", float("nan")), ".2f"),
            _fmt_mean_std(m.get("cd", {}).get("mean", float("nan")), m.get("cd", {}).get("std", float("nan")), ".4f"),
        ])

    # Markdown
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("|" + "|".join(["---"] * len(headers)) + "|\n")
        for r in rows:
            f.write("| " + " | ".join(r) + " |\n")

    # LaTeX (simple)
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write("\\begin{table}[t]\n\\centering\n")
        f.write("\\begin{tabular}{lrrrrrrrrr}\\hline\n")
        f.write("Method & Time$\\downarrow$ & Speed$\\uparrow$ & Div$\\uparrow$ & Temp$\\uparrow$ & Phys$\\uparrow$ & mIoU$\\uparrow$ & FID$\\downarrow$ & FVD$\\downarrow$ & CD$\\downarrow$\\\\\\hline\n")
        for r in rows:
            rr = [r[0]] + [c.replace("±", "\\\\pm") for c in r[1:]]
            f.write("{} & {} & {} & {} & {} & {} & {} & {} & {} & {}\\\\\n".format(*rr))
        f.write("\\hline\\end{tabular}\n")
        f.write("\\caption{OccSora architecture ablation.}\n")
        f.write("\\end{table}\n")


# -------------------------
# Main
# -------------------------


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, default="DiT-STCA-XL/2", choices=list(DiT_STCA_models.keys()))
    parser.add_argument("--num-samples", type=int, default=4, help="samples per method")
    parser.add_argument("--num-steps", type=int, default=50)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--cond-indices", type=str, default="0", help="comma-separated condition indices, e.g. 0,1,2")
    parser.add_argument("--cond-idx", type=int, default=None, help="(deprecated) single condition index")
    parser.add_argument("--gt-mode-dir", type=str, default="/root/autodl-tmp/OccSora_output/vqvae/step32-2/gt_mode")
    parser.add_argument("--token-dir", type=str, default="/root/autodl-tmp/OccSora_output/vqvae/step32-2/token")

    parser.add_argument("--vqvae-ckpt", type=str, default="/root/autodl-tmp/OccSoraModel/epoch_125.pth")
    parser.add_argument("--vqvae-config", type=str, default="/root/OccSora-main/config/train_vqvae.py")

    parser.add_argument("--out-dir", type=str, default="/root/OccSora-main/eval_results")

    parser.add_argument("--skip-occupancy", action="store_true", help="skip mIoU/CD/physics decoding")
    parser.add_argument("--decode-per-cond", type=int, default=1, help="decode this many generated samples per condition (to bound runtime)")
    parser.add_argument("--cd-frame-stride", type=int, default=4)
    parser.add_argument("--cd-max-points", type=int, default=15000)

    # 自定义 checkpoint 路径（可选，用于评估用户自己训练的模型）
    parser.add_argument("--baseline-ckpt", type=str, default=None, help="Custom baseline model checkpoint path")
    parser.add_argument("--stca-ckpt", type=str, default=None, help="Custom STCA model checkpoint path")
    parser.add_argument("--sads-ckpt", type=str, default=None, help="Custom SADS model checkpoint path")
    parser.add_argument("--full-ckpt", type=str, default=None, help="Custom full model checkpoint path")

    args = parser.parse_args()

    def _parse_cond_indices(s: str) -> List[int]:
        s = (s or "").strip()
        if not s:
            return []
        out: List[int] = []
        for part in s.split(","):
            part = part.strip()
            if not part:
                continue
            if "-" in part:
                a, b = part.split("-", 1)
                out.extend(list(range(int(a), int(b) + 1)))
            else:
                out.append(int(part))
        return out

    cond_indices = _parse_cond_indices(args.cond_indices)
    if args.cond_idx is not None:
        cond_indices = [int(args.cond_idx)]
    if not cond_indices:
        raise ValueError("No condition indices provided")

    ensure_dir(args.out_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model checkpoints (优先使用用户指定的路径，否则使用默认fallback)
    baseline_ckpt = args.baseline_ckpt if args.baseline_ckpt and os.path.exists(args.baseline_ckpt) else pick_ckpt(
        f"/root/autodl-tmp/model/baseline/dit_stca_epoch100.pt",
        f"/root/autodl-tmp/model/baseline_real/dit_stca_epoch100.pt",
        f"/root/autodl-tmp/model/baseline_real",
        # Workspace fallbacks
        f"/root/OccSora-main/out/ablation_baseline/final.pt",
        f"/root/OccSora-main/checkpoints/baseline/dit_stca_epoch100.pt",
        f"/root/OccSora-main/checkpoints/baseline",
    )
    stca_ckpt = args.stca_ckpt if args.stca_ckpt and os.path.exists(args.stca_ckpt) else pick_ckpt(
        f"/root/autodl-tmp/model/stca_real/dit_stca_epoch100.pt",
        f"/root/autodl-tmp/model/stca_only/dit_stca_epoch100.pt",
        f"/root/autodl-tmp/model/stca_only",
        f"/root/autodl-tmp/model/stca_real",
        # Workspace fallbacks
        f"/root/OccSora-main/checkpoints/stca_only/dit_stca_epoch100.pt",
        f"/root/OccSora-main/checkpoints/stca_only",
    )
    sads_ckpt = args.sads_ckpt if args.sads_ckpt and os.path.exists(args.sads_ckpt) else pick_ckpt(
        f"/root/autodl-tmp/model/sads_only/dit_stca_epoch100.pt",
        f"/root/autodl-tmp/model/sads_500/dit_stca_epoch100.pt",
        f"/root/autodl-tmp/model/sads_only",
        f"/root/autodl-tmp/model/sads_500",
    )
    full_ckpt = args.full_ckpt if args.full_ckpt and os.path.exists(args.full_ckpt) else pick_ckpt(
        f"/root/autodl-tmp/model/full_innovation/dit_stca_epoch100.pt",
        f"/root/autodl-tmp/model/full_innovation",
        # Workspace fallbacks
        f"/root/OccSora-main/out/occsora_paper_full/final.pt",
        f"/root/OccSora-main/out/occsora_plus_paper/final.pt",
    )

    specs = [
        ModelSpec("baseline", "Baseline", baseline_ckpt, use_stca=False, use_sads=False),
        ModelSpec("stca", "+STCA", stca_ckpt, use_stca=True, use_sads=False),
        ModelSpec("sads", "+SADS", sads_ckpt, use_stca=True, use_sads=True),
        ModelSpec("full", "Full", full_ckpt, use_stca=True, use_sads=True),
    ]

    # Real distribution for FID/FVD
    real_latents = load_real_latents(args.token_dir, num_samples=max(args.num_samples, 16))

    # Feature extractors
    feat2d = FeatureExtractor2D().to(device).eval()
    feat3d = FeatureExtractor3D().to(device).eval()

    with torch.no_grad():
        real_2d = feat2d(real_latents.to(device).mean(dim=2))
        real_3d = feat3d(real_latents.to(device))
    real_2d_np = real_2d.detach().cpu().numpy()
    real_3d_np = real_3d.detach().cpu().numpy()

    vq = None
    ref_occ_by_cond: Dict[int, np.ndarray] = {}
    empty_label = None
    num_classes = None
    if not args.skip_occupancy:
        if not os.path.exists(args.vqvae_ckpt):
            raise FileNotFoundError(f"VQVAE checkpoint not found: {args.vqvae_ckpt}")
        vq = load_vqvae(args.vqvae_ckpt, args.vqvae_config, device)
        num_classes = int(getattr(vq, "num_classes", 18))
        empty_label = int(num_classes - 1)
        for cond_idx in cond_indices:
            ref_token = np.load(os.path.join(args.token_dir, f"i_iter_{cond_idx}.npy")).astype(np.float32)
            ref_token_128 = np.concatenate([ref_token, ref_token], axis=0)[None, ...]  # (1,128,4,25,25)
            ref_occ_t = decode_latent_to_semantic(vq, torch.from_numpy(ref_token_128).to(device))
            ref_occ_by_cond[cond_idx] = ref_occ_t[0].numpy()  # (F,H,W,D)

    results_by_cond: Dict[str, Dict[int, Dict[str, float]]] = {}
    summary: Dict[str, Dict[str, Dict[str, float]]] = {}

    for spec in specs:
        if not spec.ckpt:
            raise FileNotFoundError(f"No checkpoint found for {spec.key}")

        print(f"\n=== Evaluating {spec.key} ===")
        print(f"ckpt: {spec.ckpt}")

        model = DiT_STCA_models[args.model](use_stca=spec.use_stca).to(device)
        _load_ckpt_into_model(model, spec.ckpt, device)
        model.eval()

        results_by_cond[spec.key] = {}
        for cond_idx in cond_indices:
            condition_vec = load_condition_vec(cond_idx, args.gt_mode_dir)

            samples_list: List[np.ndarray] = []
            times: List[float] = []
            for i in range(args.num_samples):
                s, t_sec, _ = sample_one(
                    model=model,
                    device=device,
                    condition_vec=condition_vec,
                    seed=args.seed + i,
                    num_steps=args.num_steps,
                    cfg_scale=args.cfg_scale,
                    use_sads=spec.use_sads,
                )
                samples_list.append(s)
                times.append(t_sec)

            samples = np.concatenate(samples_list, axis=0)  # (N,128,4,25,25)
            np.save(os.path.join(args.out_dir, f"samples_{spec.key}_cond{cond_idx}.npy"), samples)

            avg_time = float(np.mean(times))
            throughput = float(np.mean([1.0 / t for t in times if t > 0])) if times else 0.0

            div = diversity_rmse(samples)
            tc = temporal_consistency_latent(samples, device)

            with torch.no_grad():
                fake_2d = feat2d(torch.from_numpy(samples).to(device).mean(dim=2))
                fake_3d = feat3d(torch.from_numpy(samples).to(device))
            fid = frechet_distance(real_2d_np, fake_2d.detach().cpu().numpy())
            fvd = frechet_distance(real_3d_np, fake_3d.detach().cpu().numpy())

            metrics: Dict[str, float] = {
                "avg_time": avg_time,
                "throughput": throughput,
                "diversity": div,
                **tc,
                "fid": float(fid),
                "fvd": float(fvd),
            }

            if not args.skip_occupancy:
                assert vq is not None
                assert empty_label is not None and num_classes is not None
                ref_occ = ref_occ_by_cond[cond_idx]

                decode_n = max(1, min(int(args.decode_per_cond), samples.shape[0]))
                miou_list: List[float] = []
                cd_list: List[float] = []
                phys_score_list: List[float] = []
                phys_occ_ratio_list: List[float] = []
                phys_occ_match_list: List[float] = []
                phys_ground_list: List[float] = []
                phys_unsupported_list: List[float] = []
                phys_motion_list: List[float] = []
                phys_accel_list: List[float] = []

                for j in range(decode_n):
                    pred_latent = torch.from_numpy(samples[j : j + 1]).to(device)
                    pred_occ_t = decode_latent_to_semantic(vq, pred_latent)
                    pred_occ = pred_occ_t[0].numpy()

                    occ_m = occupancy_metrics(
                        pred_occ,
                        ref_occ,
                        empty_label=empty_label,
                        num_classes=num_classes,
                        cd_frame_stride=args.cd_frame_stride,
                        cd_max_points=args.cd_max_points,
                    )
                    phys_m = physics_consistency_occupancy(pred_occ, empty_label=empty_label, ref_occ=ref_occ)

                    miou_list.append(float(occ_m.get("miou", float("nan"))))
                    cd_list.append(float(occ_m.get("cd", float("nan"))))
                    phys_score_list.append(float(phys_m.get("physics_score", float("nan"))))
                    phys_occ_ratio_list.append(float(phys_m.get("occ_ratio", float("nan"))))
                    phys_occ_match_list.append(float(phys_m.get("occupancy_match_score", float("nan"))))
                    phys_ground_list.append(float(phys_m.get("ground_support_score", float("nan"))))
                    phys_unsupported_list.append(float(phys_m.get("unsupported_ratio", float("nan"))))
                    phys_motion_list.append(float(phys_m.get("motion_smooth_score", float("nan"))))
                    phys_accel_list.append(float(phys_m.get("motion_accel", float("nan"))))

                metrics.update({
                    "miou": float(np.nanmean(miou_list)) if miou_list else float("nan"),
                    "cd": float(np.nanmean(cd_list)) if cd_list else float("nan"),
                    "physics_score": float(np.nanmean(phys_score_list)) if phys_score_list else float("nan"),
                    "physics_occ_ratio": float(np.nanmean(phys_occ_ratio_list)) if phys_occ_ratio_list else float("nan"),
                    "physics_occ_match": float(np.nanmean(phys_occ_match_list)) if phys_occ_match_list else float("nan"),
                    "physics_ground_support": float(np.nanmean(phys_ground_list)) if phys_ground_list else float("nan"),
                    "physics_unsupported_ratio": float(np.nanmean(phys_unsupported_list)) if phys_unsupported_list else float("nan"),
                    "physics_motion_smooth": float(np.nanmean(phys_motion_list)) if phys_motion_list else float("nan"),
                    "physics_motion_accel": float(np.nanmean(phys_accel_list)) if phys_accel_list else float("nan"),
                })
            else:
                metrics.update({
                    "miou": float("nan"),
                    "cd": float("nan"),
                    "physics_score": float("nan"),
                    "physics_occ_ratio": float("nan"),
                    "physics_occ_match": float("nan"),
                    "physics_ground_support": float("nan"),
                    "physics_unsupported_ratio": float("nan"),
                    "physics_motion_smooth": float("nan"),
                    "physics_motion_accel": float("nan"),
                })

            results_by_cond[spec.key][cond_idx] = metrics

        # Summary (mean±std across conditions)
        metric_keys = sorted({k for v in results_by_cond[spec.key].values() for k in v.keys()})
        summary[spec.key] = {}
        for mk in metric_keys:
            vals = [float(results_by_cond[spec.key][ci].get(mk, float("nan"))) for ci in cond_indices]
            vals = [v for v in vals if not np.isnan(v)]
            mu, sd = mean_std(vals)
            summary[spec.key][mk] = {"mean": mu, "std": sd}

    # Persist
    out_by_cond = os.path.join(args.out_dir, "metrics_by_cond.json")
    with open(out_by_cond, "w", encoding="utf-8") as f:
        json.dump(results_by_cond, f, indent=2)

    out_summary = os.path.join(args.out_dir, "metrics_summary.json")
    with open(out_summary, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Back-compat: keep metrics.json as summary
    out_json = os.path.join(args.out_dir, "metrics.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    write_tables(summary, args.out_dir)

    print("\nSaved:")
    print(f"  {out_json}")
    print(f"  {out_by_cond}")
    print(f"  {out_summary}")
    print(f"  {os.path.join(args.out_dir, 'metrics_table.md')}")
    print(f"  {os.path.join(args.out_dir, 'metrics_table.tex')}")


if __name__ == "__main__":
    main()
