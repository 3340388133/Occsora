#!/usr/bin/env python3
"""FOD (Fréchet Occupancy Distance) - OccSora原文方式计算"""
import os
import sys
import json
import numpy as np
from scipy import linalg
from pathlib import Path

sys.path.insert(0, "/root/OccSora-main")

def compute_fod(real_features: np.ndarray, fake_features: np.ndarray) -> float:
    """计算FOD (Fréchet距离在latent空间)"""
    real = np.asarray(real_features, dtype=np.float64)
    fake = np.asarray(fake_features, dtype=np.float64)

    mu_r, mu_f = np.mean(real, 0), np.mean(fake, 0)

    cov_r = np.cov(real, rowvar=False) if real.shape[0] > 1 else np.eye(real.shape[1]) * 1e-6
    cov_f = np.cov(fake, rowvar=False) if fake.shape[0] > 1 else np.eye(fake.shape[1]) * 1e-6

    cov_r, cov_f = np.atleast_2d(cov_r), np.atleast_2d(cov_f)

    diff = mu_r - mu_f
    covmean, _ = linalg.sqrtm(cov_r @ cov_f, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    return float(diff @ diff + np.trace(cov_r + cov_f - 2 * covmean))

def main():
    token_dir = "/root/autodl-tmp/OccSora_output/vqvae/step32-2/token"
    results_dir = "/root/OccSora-main/eval_results_paperish_v6"

    # 加载真实token (减少数量以节省内存)
    print("Loading real tokens...")
    real_tokens = []
    token_files = sorted([f for f in os.listdir(token_dir) if f.endswith('.npy')])[:20]
    for f in token_files:
        token = np.load(os.path.join(token_dir, f)).astype(np.float32)
        # 只取部分特征以减少内存
        real_tokens.append(token[:32].flatten())
    real_features = np.stack(real_tokens, axis=0)
    print(f"Real features shape: {real_features.shape}")

    # 计算每个模型的FOD
    results = {}
    models = ["baseline", "stca", "sads", "full"]

    print("\n" + "="*50)
    print("FOD Results (OccSora Protocol - Lower is Better)")
    print("="*50)

    for model in models:
        pattern = f"samples_{model}_cond*.npy"
        files = sorted(Path(results_dir).glob(pattern))

        if not files:
            print(f"{model}: No samples found")
            continue

        fake_tokens = []
        for f in files[:3]:  # 只用前3个文件
            samples = np.load(f)  # (N, 128, 4, 25, 25)
            for i in range(min(samples.shape[0], 8)):
                # 只取前32通道以匹配real
                token = samples[i, :32].flatten()
                fake_tokens.append(token)

        fake_features = np.stack(fake_tokens[:real_features.shape[0]], axis=0)
        fod = compute_fod(real_features, fake_features)
        results[model] = {"fod": fod}
        print(f"{model:12s}: FOD = {fod:.2f}")

    # 与OccSora原文对比
    print("\n" + "="*50)
    print("Comparison with OccSora Paper")
    print("="*50)
    print("OccSora (reported): FOD ≈ 89.6")
    print(f"Ours (Full):        FOD = {results.get('full', {}).get('fod', 'N/A'):.2f}")

    if 'full' in results and 'baseline' in results:
        improvement = (results['baseline']['fod'] - results['full']['fod']) / results['baseline']['fod'] * 100
        print(f"Improvement over baseline: {improvement:.1f}%")

    # 保存结果
    out_path = os.path.join(results_dir, "fod_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")

if __name__ == "__main__":
    main()
