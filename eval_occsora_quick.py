#!/usr/bin/env python3
"""评估OccSora模型 - 与GT对比"""

import os
os.environ['HOME'] = '/root'

import sys
sys.path.insert(0, '/root/OccSora-main')

import numpy as np
import torch
from tqdm import tqdm
import json

from mmengine import Config
from mmengine.registry import MODELS
import model as model_module


def decode_latent(latent, vqvae_model, device):
    """解码单个latent到occupancy"""
    latent = latent * 10
    z = torch.from_numpy(latent[:, :64]).float().to(device)

    with torch.no_grad():
        x = vqvae_model.post_vq_conv(z)
        x = vqvae_model.decoder_gpt(x)
        x = x.squeeze(0)

        F_out, H, W = x.shape[1], x.shape[2], x.shape[3]
        D = 16
        expansion = vqvae_model.expansion

        logits_raw = x.permute(1, 2, 3, 0)
        logits_raw = logits_raw.reshape(F_out, H, W, D, expansion)

        template = vqvae_model.class_embeds.weight.T.unsqueeze(0)
        logits_flat = logits_raw.reshape(-1, D, expansion)
        similarity = torch.matmul(logits_flat, template)
        logits = similarity.reshape(1, F_out, H, W, D, vqvae_model.num_cls)

        pred = logits.argmax(dim=-1).squeeze(0).cpu().numpy()

    return pred  # (F, H, W, D)


def compute_iou(pred, gt, num_classes=18):
    """计算IoU"""
    ious = []
    for c in range(1, num_classes):  # 跳过empty
        pred_c = (pred == c)
        gt_c = (gt == c)
        intersection = (pred_c & gt_c).sum()
        union = (pred_c | gt_c).sum()
        if union > 0:
            ious.append(intersection / union)
    return np.mean(ious) if ious else 0.0


def main():
    device = "cuda"
    print("Loading VQVAE...")

    cfg = Config.fromfile('/root/OccSora-main/config/train_vqvae.py')
    vqvae = MODELS.build(cfg.model)
    ckpt = torch.load('/root/autodl-tmp/OccSoraModel/epoch_125.pth', map_location='cpu')
    vqvae.load_state_dict(ckpt['state_dict'], strict=False)
    vqvae = vqvae.to(device).eval()

    # 加载GT tokens
    gt_dir = '/root/OccSora-main/out/gt_mode_occstats'

    results = {'occsora_full': [], 'occsora_baseline': []}

    for cond_idx in range(8):
        print(f"\nProcessing condition {cond_idx}...")

        # 加载OccSora Full样本
        full_path = f'/root/OccSora-main/eval_results_paperish_v6/samples_full_cond{cond_idx}.npy'
        baseline_path = f'/root/OccSora-main/eval_results_paperish_v6/samples_baseline_cond{cond_idx}.npy'

        if os.path.exists(full_path):
            latent = np.load(full_path)
            pred = decode_latent(latent[:1], vqvae, device)
            print(f"  Full pred shape: {pred.shape}")

            # 计算统计
            occupied = (pred > 0) & (pred < 17)
            print(f"  Occupied voxels: {occupied.sum()}")
            results['occsora_full'].append({
                'cond': cond_idx,
                'occupied_ratio': float(occupied.mean()),
                'pred_shape': list(pred.shape)
            })

        if os.path.exists(baseline_path):
            latent = np.load(baseline_path)
            pred = decode_latent(latent[:1], vqvae, device)
            occupied = (pred > 0) & (pred < 17)
            results['occsora_baseline'].append({
                'cond': cond_idx,
                'occupied_ratio': float(occupied.mean()),
            })

    # 保存结果
    with open('/root/OccSora-main/occsora_eval_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print("\n" + "="*50)
    print("OccSora Evaluation Results")
    print("="*50)

    full_occ = np.mean([r['occupied_ratio'] for r in results['occsora_full']])
    baseline_occ = np.mean([r['occupied_ratio'] for r in results['occsora_baseline']])

    print(f"Full model avg occupied ratio: {full_occ*100:.2f}%")
    print(f"Baseline avg occupied ratio: {baseline_occ*100:.2f}%")


if __name__ == "__main__":
    main()
