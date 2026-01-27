#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
import pickle
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np


VEHICLE_CLASSES_16 = {2, 3, 4, 5, 6, 9, 10}
PEDESTRIAN_CLASSES_16 = {7}
EMPTY_CLASS_16 = 17


@dataclass
class SceneStat:
    scene: str
    occ_ratio_mean: float
    veh_ratio_mean: float
    ped_ratio_mean: float


def _load_infos(infos_pkl: str) -> Dict[str, List[dict]]:
    with open(infos_pkl, "rb") as f:
        data = pickle.load(f)
    infos = data.get("infos")
    if not isinstance(infos, dict) or not infos:
        raise ValueError(f"Invalid infos pkl: {infos_pkl}")
    return infos


def _label_path(data_path: str, dataset: str, scene: str, token: str) -> str:
    return os.path.join(data_path, dataset, scene, token, "labels.npz")


def _frame_stats(sem: np.ndarray) -> Tuple[float, float, float]:
    sem = np.asarray(sem)
    total = sem.size
    if total == 0:
        return 0.0, 0.0, 0.0

    non_empty = (sem != EMPTY_CLASS_16)
    occ = float(non_empty.sum())

    veh = float(np.isin(sem, list(VEHICLE_CLASSES_16)).sum())
    ped = float(np.isin(sem, list(PEDESTRIAN_CLASSES_16)).sum())

    return occ / total, veh / total, ped / total


def _frame_score(occ_ratio: float, veh_ratio: float, ped_ratio: float) -> float:
    # A single scalar that increases with traffic complexity.
    # Keep within a small numeric range similar to original 0..3.
    score = 3.0 * (
        0.60 * veh_ratio +
        0.25 * ped_ratio +
        0.15 * occ_ratio
    )
    return float(score)


def compute_scene_stats(infos: Dict[str, List[dict]], data_path: str, dataset: str, max_frames: int) -> List[SceneStat]:
    out: List[SceneStat] = []
    for scene, seq in infos.items():
        occs: List[float] = []
        vehs: List[float] = []
        peds: List[float] = []
        for fr in seq[:max_frames] if max_frames > 0 else seq:
            token = fr.get("token")
            if not token:
                continue
            lp = _label_path(data_path, dataset, scene, token)
            if not os.path.exists(lp):
                continue
            sem = np.load(lp)["semantics"]
            occ, veh, ped = _frame_stats(sem)
            occs.append(occ)
            vehs.append(veh)
            peds.append(ped)
        if not occs:
            continue
        out.append(SceneStat(
            scene=scene,
            occ_ratio_mean=float(np.mean(occs)),
            veh_ratio_mean=float(np.mean(vehs)),
            ped_ratio_mean=float(np.mean(peds)),
        ))
    return out


def pick_diverse_scenes(stats: List[SceneStat], k: int) -> List[SceneStat]:
    if k <= 0:
        return []
    if len(stats) <= k:
        return stats

    # Sort by a combined complexity metric, then pick evenly spaced.
    stats = sorted(stats, key=lambda s: (0.7 * s.veh_ratio_mean + 0.3 * s.ped_ratio_mean, s.occ_ratio_mean))
    idxs = np.linspace(0, len(stats) - 1, k).round().astype(int)
    return [stats[i] for i in idxs]


def build_condition_for_scene(
    infos: Dict[str, List[dict]],
    scene: str,
    data_path: str,
    dataset: str,
    max_frames: int,
) -> np.ndarray:
    seq = infos[scene]
    scores: List[float] = []
    frames = seq[:max_frames] if max_frames > 0 else seq
    for fr in frames:
        token = fr.get("token")
        if not token:
            scores.append(0.0)
            continue
        lp = _label_path(data_path, dataset, scene, token)
        if not os.path.exists(lp):
            scores.append(0.0)
            continue
        sem = np.load(lp)["semantics"]
        occ, veh, ped = _frame_stats(sem)
        scores.append(_frame_score(occ, veh, ped))

    v = np.asarray(scores, dtype=np.float32)
    if v.size == 0:
        return np.zeros((1,), dtype=np.float32)
    return v


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate gt_mode/*.npy conditions from occupancy label statistics (no nuScenes tables needed)")
    ap.add_argument("--infos-pkl", default="data/nuscenes_infos_val_temporal_v3_scene.pkl")
    ap.add_argument("--data-path", default="data/nuscenes")
    ap.add_argument("--dataset", default="gts", choices=["gts", "tpv_dense", "tpv_sparse"])
    ap.add_argument("--out-dir", required=True, help="Output directory for i_iter_*.npy")
    ap.add_argument("--num-conds", type=int, default=8, help="How many condition files to generate")
    ap.add_argument("--max-frames", type=int, default=31, help="Frames per condition vector (will be padded to 64 at runtime)")
    args = ap.parse_args()

    infos = _load_infos(args.infos_pkl)

    stats = compute_scene_stats(infos, args.data_path, args.dataset, args.max_frames)
    if not stats:
        raise RuntimeError(
            "No scenes had readable labels. Check --data-path/--dataset and that labels.npz exists under e.g. data/nuscenes/gts/<scene>/<token>/labels.npz"
        )

    chosen = pick_diverse_scenes(stats, args.num_conds)
    os.makedirs(args.out_dir, exist_ok=True)

    for i, st in enumerate(chosen):
        v = build_condition_for_scene(infos, st.scene, args.data_path, args.dataset, args.max_frames)
        out = os.path.join(args.out_dir, f"i_iter_{i}.npy")
        np.save(out, v.astype(np.float32))
        print(
            f"[{i}] scene={st.scene} frames={len(v)} occ={st.occ_ratio_mean:.4f} veh={st.veh_ratio_mean:.4f} ped={st.ped_ratio_mean:.4f} -> {out}"
        )


if __name__ == "__main__":
    main()
