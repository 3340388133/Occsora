#!/usr/bin/env python3
"""消融实验 - CFG scale和训练epoch影响分析"""
import os
import sys
import json
import numpy as np
import torch
from pathlib import Path
from scipy import linalg

sys.path.insert(0, "/root/OccSora-main")

from models_stca import DiT_STCA_models
from diffusion import create_diffusion

def compute_fod(real, fake):
    """快速FOD计算"""
    mu_r, mu_f = np.mean(real, 0), np.mean(fake, 0)
    cov_r = np.cov(real, rowvar=False) if real.shape[0] > 1 else np.eye(real.shape[1]) * 1e-6
    cov_f = np.cov(fake, rowvar=False) if fake.shape[0] > 1 else np.eye(fake.shape[1]) * 1e-6
    diff = mu_r - mu_f
    covmean, _ = linalg.sqrtm(np.atleast_2d(cov_r) @ np.atleast_2d(cov_f), disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff @ diff + np.trace(cov_r + cov_f - 2 * covmean))

def compute_diversity(samples):
    """计算多样性"""
    n = samples.shape[0]
    if n < 2:
        return 0.0
    flat = samples.reshape(n, -1).astype(np.float64)
    dists = []
    for _ in range(min(50, n*(n-1)//2)):
        i, j = np.random.choice(n, 2, replace=False)
        dists.append(np.sqrt(np.mean((flat[i] - flat[j])**2)))
    return float(np.mean(dists))

def sample_batch(model, diffusion, device, cond, n_samples=4, cfg_scale=4.0):
    """生成样本"""
    samples = []
    for i in range(n_samples):
        torch.manual_seed(42 + i)
        y = torch.tensor(np.stack([cond, cond], 0).astype(np.float32), device=device)
        z = torch.randn((2, 128, 4, 25, 25), device=device)
        with torch.no_grad():
            out = diffusion.p_sample_loop(
                model.forward_with_cfg,
                z.shape, z,
                clip_denoised=False,
                model_kwargs=dict(y=y, cfg_scale=cfg_scale),
                progress=False, device=device
            )
        out, _ = out.chunk(2, dim=0)
        samples.append(out.cpu().numpy())
    return np.concatenate(samples, axis=0)

def main():
    output_dir = "/root/OccSora-main/ablation_results"
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载条件
    cond_path = "/root/autodl-tmp/OccSora_output/vqvae/step32-2/gt_mode/i_iter_0.npy"
    cond = np.load(cond_path).astype(np.float32)
    if len(cond) < 64:
        cond = np.pad(cond, (0, 64 - len(cond)))
    else:
        cond = cond[:64]

    # 加载真实token作为参考
    token_dir = "/root/autodl-tmp/OccSora_output/vqvae/step32-2/token"
    real_tokens = []
    for f in sorted(os.listdir(token_dir))[:50]:
        if f.endswith('.npy'):
            real_tokens.append(np.load(os.path.join(token_dir, f)).flatten())
    real_features = np.stack(real_tokens, axis=0)

    results = {"cfg_ablation": {}, "epoch_ablation": {}}

    # ============ CFG Scale消融 ============
    print("\n" + "="*50)
    print("CFG Scale Ablation")
    print("="*50)

    ckpt_path = "/root/autodl-tmp/model/full_innovation/dit_stca_epoch100.pt"
    if os.path.exists(ckpt_path):
        model = DiT_STCA_models["DiT-STCA-XL/2"](use_stca=True).to(device)
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=False)
        model.eval()
        diffusion = create_diffusion("50")

        cfg_scales = [1.0, 2.0, 4.0, 6.0, 8.0]
        print(f"{'CFG':>6} {'FOD':>10} {'Diversity':>12}")
        print("-" * 30)

        for cfg in cfg_scales:
            samples = sample_batch(model, diffusion, device, cond, n_samples=4, cfg_scale=cfg)
            fake_features = samples[:, :64].reshape(samples.shape[0], -1)
            fod = compute_fod(real_features, fake_features)
            div = compute_diversity(samples)
            results["cfg_ablation"][str(cfg)] = {"fod": fod, "diversity": div}
            print(f"{cfg:>6.1f} {fod:>10.2f} {div:>12.2f}")
    else:
        print(f"Checkpoint not found: {ckpt_path}")

    # ============ Epoch消融 ============
    print("\n" + "="*50)
    print("Training Epoch Ablation")
    print("="*50)

    ckpt_dir = "/root/autodl-tmp/model/full_innovation"
    epochs = [20, 40, 60, 80, 100]

    print(f"{'Epoch':>6} {'FOD':>10} {'Diversity':>12}")
    print("-" * 30)

    for epoch in epochs:
        ckpt_path = f"{ckpt_dir}/dit_stca_epoch{epoch}.pt"
        if not os.path.exists(ckpt_path):
            # 尝试其他命名
            alt_paths = [
                f"{ckpt_dir}/epoch{epoch}.pt",
                f"{ckpt_dir}/checkpoint_{epoch}.pt",
            ]
            for alt in alt_paths:
                if os.path.exists(alt):
                    ckpt_path = alt
                    break

        if os.path.exists(ckpt_path):
            model = DiT_STCA_models["DiT-STCA-XL/2"](use_stca=True).to(device)
            ckpt = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=False)
            model.eval()
            diffusion = create_diffusion("50")

            samples = sample_batch(model, diffusion, device, cond, n_samples=4, cfg_scale=4.0)
            fake_features = samples[:, :64].reshape(samples.shape[0], -1)
            fod = compute_fod(real_features, fake_features)
            div = compute_diversity(samples)
            results["epoch_ablation"][str(epoch)] = {"fod": fod, "diversity": div}
            print(f"{epoch:>6} {fod:>10.2f} {div:>12.2f}")
        else:
            print(f"{epoch:>6} {'N/A':>10} {'N/A':>12} (checkpoint not found)")

    # 保存结果
    out_path = f"{output_dir}/ablation_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # 生成表格
    print("\n" + "="*50)
    print("LaTeX Table")
    print("="*50)

    print(r"\begin{table}[t]")
    print(r"\centering")
    print(r"\caption{Ablation Study on CFG Scale and Training Epochs}")
    print(r"\begin{tabular}{lcc}")
    print(r"\hline")
    print(r"Setting & FOD$\downarrow$ & Diversity$\uparrow$ \\")
    print(r"\hline")
    print(r"\multicolumn{3}{l}{\textit{CFG Scale (epoch=100)}} \\")
    for cfg, v in results["cfg_ablation"].items():
        print(f"CFG={cfg} & {v['fod']:.2f} & {v['diversity']:.2f} \\\\")
    print(r"\hline")
    print(r"\multicolumn{3}{l}{\textit{Training Epochs (CFG=4.0)}} \\")
    for ep, v in results["epoch_ablation"].items():
        print(f"Epoch {ep} & {v['fod']:.2f} & {v['diversity']:.2f} \\\\")
    print(r"\hline")
    print(r"\end{tabular}")
    print(r"\end{table}")

if __name__ == "__main__":
    main()
