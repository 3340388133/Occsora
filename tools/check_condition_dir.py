#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import glob
import os
from typing import List

import numpy as np


def _load_vec(path: str) -> np.ndarray:
    v = np.load(path).astype(np.float32)
    if v.shape[0] < 64:
        rep = (64 // v.shape[0]) + 1
        v = np.tile(v, rep)[:64]
    else:
        v = v[:64]
    return v


def main() -> None:
    ap = argparse.ArgumentParser(description="Sanity-check gt_mode/*.npy condition vectors")
    ap.add_argument("--gt-mode-dir", required=True)
    args = ap.parse_args()

    pat = os.path.join(args.gt_mode_dir, "i_iter_*.npy")
    files: List[str] = sorted(glob.glob(pat), key=lambda p: int(os.path.splitext(os.path.basename(p))[0].split("_")[-1]))
    if not files:
        raise SystemExit(f"No files matched: {pat}")

    vecs = []
    for p in files:
        v = _load_vec(p)
        vecs.append(v)
        nz = float((v != 0).mean())
        print(f"{os.path.basename(p):>14s} shape={v.shape} min={v.min():.4f} mean={v.mean():.4f} max={v.max():.4f} nonzero={nz:.2%}")

    if len(vecs) >= 2:
        print("\nPairwise L2 distance (after pad/trunc to 64):")
        for i in range(len(vecs)):
            for j in range(i + 1, len(vecs)):
                d = float(np.linalg.norm(vecs[i] - vecs[j]))
                print(f"  {i} vs {j}: {d:.6f}")


if __name__ == "__main__":
    main()
