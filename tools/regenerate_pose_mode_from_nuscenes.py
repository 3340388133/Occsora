#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
import pickle
import time
from typing import Dict, List, Tuple

import numpy as np
from pyquaternion import Quaternion


def _wrap_to_pi(a: float) -> float:
    return (a + np.pi) % (2 * np.pi) - np.pi


def _yaw_from_quat(qwxyz_or_xyzw) -> float:
    q = Quaternion(qwxyz_or_xyzw)
    yaw, _, _ = q.yaw_pitch_roll
    return float(yaw)


def _get_ego_pose_for_sample_token(nusc, sample_token: str) -> Tuple[np.ndarray, float, float]:
    """Return (pos_xyz, yaw, timestamp_sec) for a sample token via LIDAR_TOP ego_pose."""
    sample = nusc.get("sample", sample_token)
    sd_token = sample["data"]["LIDAR_TOP"]
    sd = nusc.get("sample_data", sd_token)
    ego_pose = nusc.get("ego_pose", sd["ego_pose_token"])

    pos = np.asarray(ego_pose["translation"], dtype=np.float32)
    yaw = _yaw_from_quat(ego_pose["rotation"])
    ts = float(sample["timestamp"]) * 1e-6
    return pos, yaw, ts


def _classify_mode(
    pos0: np.ndarray,
    yaw0: float,
    t0: float,
    pos1: np.ndarray,
    yaw1: float,
    t1: float,
    stop_speed_thr: float,
    yaw_rate_thr: float,
) -> int:
    dt = max(1e-3, t1 - t0)
    dist = float(np.linalg.norm(pos1[:2] - pos0[:2]))
    speed = dist / dt
    dyaw = _wrap_to_pi(yaw1 - yaw0)
    yaw_rate = dyaw / dt

    # 0 straight, 1 left, 2 right, 3 stop
    if speed < stop_speed_thr:
        return 3
    if yaw_rate > yaw_rate_thr:
        return 1
    if yaw_rate < -yaw_rate_thr:
        return 2
    return 0


def regenerate_pose_mode(
    in_pkl: str,
    out_pkl: str,
    dataroot: str,
    version: str,
    stop_speed_thr: float,
    yaw_rate_thr: float,
    limit_scenes: int,
    dry_run: bool,
) -> Dict[str, int]:
    from nuscenes.nuscenes import NuScenes

    table_root = os.path.join(dataroot, version)
    if not os.path.exists(table_root):
        avail = []
        if os.path.isdir(dataroot):
            for name in sorted(os.listdir(dataroot)):
                if name.startswith("v1.0-") and os.path.isdir(os.path.join(dataroot, name)):
                    avail.append(name)
        msg = f"nuScenes tables not found: {table_root} (version={version})"
        if avail:
            msg += f". Available under dataroot: {', '.join(avail)}"
        raise ValueError(msg)

    with open(in_pkl, "rb") as f:
        data = pickle.load(f)
    infos = data.get("infos")
    if not isinstance(infos, dict) or not infos:
        raise ValueError(f"Invalid pkl format: missing/empty data['infos'] in {in_pkl}")

    nusc = NuScenes(version=version, dataroot=dataroot, verbose=False)

    scene_names = list(infos.keys())
    if limit_scenes > 0:
        scene_names = scene_names[:limit_scenes]

    counts = {0: 0, 1: 0, 2: 0, 3: 0}

    for si, scene_name in enumerate(scene_names):
        seq = infos[scene_name]
        if not isinstance(seq, list) or len(seq) < 2:
            continue

        poses: List[Tuple[np.ndarray, float, float]] = []
        for frame in seq:
            token = frame.get("token")
            if not token:
                poses.append((np.zeros(3, dtype=np.float32), 0.0, 0.0))
                continue
            pos, yaw, ts = _get_ego_pose_for_sample_token(nusc, token)
            poses.append((pos, yaw, ts))

        modes: List[int] = []
        for i in range(len(seq) - 1):
            pos0, yaw0, t0 = poses[i]
            pos1, yaw1, t1 = poses[i + 1]
            m = _classify_mode(pos0, yaw0, t0, pos1, yaw1, t1, stop_speed_thr, yaw_rate_thr)
            modes.append(m)
        modes.append(modes[-1])  # last frame: copy previous

        for frame, m in zip(seq, modes):
            frame["pose_mode"] = int(m)
            counts[int(m)] = counts.get(int(m), 0) + 1

        if (si + 1) % 50 == 0:
            print(f"processed {si+1}/{len(scene_names)} scenes")

    if dry_run:
        return counts

    if os.path.abspath(out_pkl) == os.path.abspath(in_pkl) and os.path.exists(in_pkl):
        ts = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        backup = in_pkl + f".bak_{ts}"
        os.rename(in_pkl, backup)
        print(f"backup created: {backup}")

    with open(out_pkl, "wb") as f:
        pickle.dump(data, f)

    return counts


def main() -> None:
    ap = argparse.ArgumentParser(description="Regenerate pose_mode in nuscenes_infos_*.pkl using nuScenes ego motion")
    ap.add_argument("--in-pkl", required=True, help="Input pkl, e.g. data/nuscenes_infos_train_temporal_v3_scene.pkl")
    ap.add_argument("--out-pkl", default=None, help="Output pkl (default: overwrite in-pkl with backup)")
    ap.add_argument("--dataroot", default="data/nuscenes", help="nuScenes dataroot")
    ap.add_argument("--version", default="v1.0-trainval", help="nuScenes version")
    ap.add_argument("--stop-speed-thr", type=float, default=0.2, help="m/s, below is STOP(3)")
    ap.add_argument("--yaw-rate-thr", type=float, default=0.15, help="rad/s, above is LEFT(1), below -thr is RIGHT(2)")
    ap.add_argument("--limit-scenes", type=int, default=0, help="For debugging; 0 means all")
    ap.add_argument("--dry-run", action="store_true", help="Compute stats but do not write")
    args = ap.parse_args()

    out_pkl = args.out_pkl or args.in_pkl

    counts = regenerate_pose_mode(
        in_pkl=args.in_pkl,
        out_pkl=out_pkl,
        dataroot=args.dataroot,
        version=args.version,
        stop_speed_thr=args.stop_speed_thr,
        yaw_rate_thr=args.yaw_rate_thr,
        limit_scenes=args.limit_scenes,
        dry_run=args.dry_run,
    )
    total = sum(counts.values())
    print("pose_mode counts:")
    for k in [0, 1, 2, 3]:
        v = counts.get(k, 0)
        r = (v / total) if total else 0.0
        print(f"  {k}: {v} ({r:.2%})")


if __name__ == "__main__":
    main()
