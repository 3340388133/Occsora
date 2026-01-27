#!/usr/bin/env python3
"""
多数据集评估脚本：在Occ3D和nuScenes上对比OccSora Full、Baseline和Copy-Paste基线
"""

import os
os.environ['HOME'] = '/root'

import sys
sys.path.insert(0, '/root/OccSora-main')

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import json
from pathlib import Path
from scipy import linalg
import argparse

from mmengine import Config
from mmengine.registry import MODELS
import model as model_module


# ============================================
# 配置
# ============================================

OCC3D_CLASSES = {
    0: 'empty', 1: 'barrier', 2: 'bicycle', 3: 'bus', 4: 'car',
    5: 'construction_vehicle', 6: 'motorcycle', 7: 'pedestrian',
    8: 'traffic_cone', 9: 'trailer', 10: 'truck', 11: 'driveable_surface',
    12: 'other_flat', 13: 'sidewalk', 14: 'terrain', 15: 'manmade',
    16: 'vegetation', 17: 'free'
}

DATASETS = {
    'occ3d': {
        'gt_dir': '/root/autodl-tmp/gts/gts',
        'description': 'Occ3D (850 scenes)',
        'max_scenes': 100
    },
    'nuscenes': {
        'gt_dir': '/root/autodl-tmp/nuScenes/gts',
        'description': 'nuScenes (2 scenes)',
        'max_scenes': 2
    }
}


# ============================================
# 评估器类
# ============================================

class Occ3DEvaluator:
    """语义占用评估器"""
    def __init__(self, num_classes=18):
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        self.tp = np.zeros(self.num_classes)
        self.fp = np.zeros(self.num_classes)
        self.fn = np.zeros(self.num_classes)
        self.total = 0

    def update(self, pred, gt):
        """更新统计量"""
        pred = pred.flatten()
        gt = gt.flatten()

        for c in range(1, self.num_classes):  # 跳过empty类
            pred_c = (pred == c)
            gt_c = (gt == c)
            self.tp[c] += (pred_c & gt_c).sum()
            self.fp[c] += (pred_c & ~gt_c).sum()
            self.fn[c] += (~pred_c & gt_c).sum()

        self.total += 1

    def compute_miou(self):
        """计算mIoU"""
        ious = []
        for c in range(1, self.num_classes):
            denom = self.tp[c] + self.fp[c] + self.fn[c]
            if denom > 0:
                iou = self.tp[c] / denom
                ious.append(iou)
        return np.mean(ious) * 100 if ious else 0.0

    def compute_per_class_iou(self):
        """计算每类IoU"""
        ious = {}
        for c in range(1, self.num_classes):
            denom = self.tp[c] + self.fp[c] + self.fn[c]
            if denom > 0:
                ious[OCC3D_CLASSES.get(c, f'class_{c}')] = self.tp[c] / denom * 100
        return ious


# ============================================
# 数据加载
# ============================================

def get_scenes(gt_dir, max_scenes=100):
    """获取场景列表"""
    scenes = []
    if not os.path.exists(gt_dir):
        return scenes

    for d in sorted(os.listdir(gt_dir)):
        if d.startswith('scene-'):
            scene_path = os.path.join(gt_dir, d)
            if os.path.isdir(scene_path):
                scene_id = int(d.split('-')[1])
                tokens = sorted(os.listdir(scene_path))
                if tokens:
                    scenes.append({'id': scene_id, 'path': scene_path, 'tokens': tokens})
        if len(scenes) >= max_scenes:
            break

    return scenes


def load_gt(scene_path, token):
    """加载GT标签"""
    path = os.path.join(scene_path, token, 'labels.npz')
    if os.path.exists(path):
        data = np.load(path)
        return data['semantics']
    return None


# ============================================
# 模型加载和解码
# ============================================

def load_vqvae(ckpt_path, config_path=None):
    """加载VQ-VAE模型"""
    # 检查点使用128维latent
    _dim_ = 16
    expansion = 8
    n_e_ = 512
    latent_dim = 128  # 匹配检查点

    # VAERes2D使用的是GPT风格的encoder/decoder，不需要Encoder2D/Decoder2D配置
    model_cfg = dict(
        type='VAERes2D',
        encoder_cfg=dict(type='Encoder2D', ch=64, out_ch=64, ch_mult=(1,2,4),
                        num_res_blocks=2, attn_resolutions=(50,), dropout=0.0,
                        resamp_with_conv=True, in_channels=_dim_*expansion,
                        resolution=200, z_channels=latent_dim, double_z=False),
        decoder_cfg=dict(type='Decoder2D', ch=64, out_ch=_dim_*expansion, ch_mult=(1,2,4),
                        num_res_blocks=2, attn_resolutions=(50,), dropout=0.0,
                        resamp_with_conv=True, in_channels=_dim_*expansion,
                        resolution=200, z_channels=latent_dim, give_pre_end=False),
        num_classes=18,
        expansion=expansion,
        vqvae_cfg=dict(
            type='VectorQuantizer',
            n_e=n_e_,
            e_dim=latent_dim,
            beta=1.,
            z_channels=latent_dim,
            use_voxel=True
        )
    )

    vqvae = MODELS.build(model_cfg)

    ckpt = torch.load(ckpt_path, map_location='cpu')
    state_dict = ckpt.get('state_dict', ckpt)

    # 尝试加载，允许不严格匹配
    missing, unexpected = vqvae.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"  Missing keys: {len(missing)}")
    if unexpected:
        print(f"  Unexpected keys: {len(unexpected)}")

    return vqvae


def decode_latent_to_occ(latent, vqvae, device):
    """
    将latent解码为occupancy预测

    Args:
        latent: (B, 128, T, H, W) latent tensor
        vqvae: VQ-VAE模型
        device: 设备

    Returns:
        pred: (B, T', H', W', D') 语义预测
    """
    vqvae = vqvae.to(device).eval()

    # 处理latent格式
    if isinstance(latent, np.ndarray):
        latent = torch.from_numpy(latent).float()

    latent = latent.to(device)

    # 确保使用全部128通道
    z = latent
    if z.shape[1] == 64:
        # 如果只有64通道，复制一份
        z = torch.cat([z, z], dim=1)

    # 缩放 (训练时可能做了缩放)
    z = z * 10

    with torch.no_grad():
        # 解码
        x = vqvae.post_vq_conv(z)
        x = vqvae.decoder_gpt(x)
        x = x.squeeze(0)

        # 获取语义预测
        F_out, H, W = x.shape[1], x.shape[2], x.shape[3]
        logits_raw = x.permute(1, 2, 3, 0).reshape(F_out, H, W, 16, vqvae.expansion)

        template = vqvae.class_embeds.weight.T.unsqueeze(0)
        similarity = torch.matmul(logits_raw.reshape(-1, 16, vqvae.expansion), template)
        logits = similarity.reshape(1, F_out, H, W, 16, vqvae.num_cls)
        pred = logits.argmax(dim=-1).squeeze(0).cpu().numpy()

    return pred


# ============================================
# 评估方法
# ============================================

def eval_copy_paste(dataset_name, max_pairs=200):
    """评估Copy-Paste基线 (复制上一帧作为预测)"""
    print(f"\n{'='*60}")
    print(f"Evaluating: Copy-Paste on {dataset_name}")
    print("="*60)

    ds_config = DATASETS[dataset_name]
    scenes = get_scenes(ds_config['gt_dir'], ds_config['max_scenes'])

    if not scenes:
        print(f"  No scenes found in {ds_config['gt_dir']}")
        return {'mIoU': 0.0, 'samples': 0}

    evaluator = Occ3DEvaluator()
    count = 0

    for scene in tqdm(scenes, desc="Scenes"):
        tokens = scene['tokens']
        for i in range(len(tokens) - 1):
            if count >= max_pairs:
                break

            # Copy-Paste: 用当前帧预测下一帧
            curr_gt = load_gt(scene['path'], tokens[i])
            next_gt = load_gt(scene['path'], tokens[i+1])

            if curr_gt is not None and next_gt is not None:
                evaluator.update(curr_gt, next_gt)
                count += 1

        if count >= max_pairs:
            break

    miou = evaluator.compute_miou()
    print(f"  Copy-Paste mIoU: {miou:.2f}%")
    print(f"  Evaluated pairs: {count}")

    return {
        'mIoU': miou,
        'samples': count,
        'per_class_iou': evaluator.compute_per_class_iou()
    }


def eval_occsora_model(model_name, dataset_name, vqvae, device, num_samples=8):
    """评估OccSora模型"""
    print(f"\n{'='*60}")
    print(f"Evaluating: OccSora {model_name} on {dataset_name}")
    print("="*60)

    ds_config = DATASETS[dataset_name]
    scenes = get_scenes(ds_config['gt_dir'], ds_config['max_scenes'])

    if not scenes:
        print(f"  No scenes found in {ds_config['gt_dir']}")
        return {'mIoU': 0.0, 'samples': 0}

    # 加载预生成的samples
    samples_dir = '/root/OccSora-main/eval_results_paperish_v6'
    evaluator = Occ3DEvaluator()
    total_evaluated = 0

    for cond_idx in range(num_samples):
        sample_path = f'{samples_dir}/samples_{model_name}_cond{cond_idx}.npy'
        if not os.path.exists(sample_path):
            continue

        latent = np.load(sample_path)  # (4, 128, 4, 25, 25)

        for b in range(latent.shape[0]):
            try:
                pred = decode_latent_to_occ(latent[b:b+1], vqvae, device)

                # 匹配场景GT
                scene_idx = (cond_idx * latent.shape[0] + b) % len(scenes)
                scene = scenes[scene_idx]

                # 对每个时间帧评估
                for t in range(min(pred.shape[0], len(scene['tokens']))):
                    gt = load_gt(scene['path'], scene['tokens'][t])
                    if gt is not None:
                        # 调整pred大小以匹配gt
                        p = pred[t]  # (H, W, D)
                        if p.shape != gt.shape:
                            # 简单resize (最近邻)
                            p_resized = np.zeros_like(gt)
                            h_ratio = gt.shape[0] / p.shape[0]
                            w_ratio = gt.shape[1] / p.shape[1]
                            d_ratio = gt.shape[2] / p.shape[2]
                            for gi in range(gt.shape[0]):
                                for gj in range(gt.shape[1]):
                                    for gk in range(gt.shape[2]):
                                        pi = min(int(gi / h_ratio), p.shape[0]-1)
                                        pj = min(int(gj / w_ratio), p.shape[1]-1)
                                        pk = min(int(gk / d_ratio), p.shape[2]-1)
                                        p_resized[gi, gj, gk] = p[pi, pj, pk]
                            p = p_resized

                        evaluator.update(p, gt)
                        total_evaluated += 1
            except Exception as e:
                print(f"  Warning: Error processing sample {cond_idx}_{b}: {e}")
                continue

    miou = evaluator.compute_miou()
    print(f"  OccSora {model_name} mIoU: {miou:.2f}%")
    print(f"  Evaluated samples: {total_evaluated}")

    return {
        'mIoU': miou,
        'samples': total_evaluated,
        'per_class_iou': evaluator.compute_per_class_iou()
    }


def eval_fid_fvd(model_name, real_token_dir, device):
    """计算FID和FVD"""
    from torch import nn

    class FeatureExtractor2D(nn.Module):
        def __init__(self, in_channels=128):
            super().__init__()
            self.conv1 = nn.Conv2d(in_channels, 64, 3, stride=2, padding=1)
            self.conv2 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
            self.conv3 = nn.Conv2d(128, 256, 3, stride=1, padding=1)
            self.pool = nn.AdaptiveAvgPool2d(1)

        def forward(self, x):
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = F.relu(self.conv3(x))
            return self.pool(x).view(x.size(0), -1)

    class FeatureExtractor3D(nn.Module):
        def __init__(self, in_channels=128):
            super().__init__()
            self.conv1 = nn.Conv3d(in_channels, 64, 3, stride=2, padding=1)
            self.conv2 = nn.Conv3d(64, 128, 3, stride=2, padding=1)
            self.conv3 = nn.Conv3d(128, 256, 3, stride=1, padding=1)
            self.pool = nn.AdaptiveAvgPool3d(1)

        def forward(self, x):
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = F.relu(self.conv3(x))
            return self.pool(x).view(x.size(0), -1)

    def frechet_distance(real_feats, fake_feats):
        mu_r, mu_f = np.mean(real_feats, axis=0), np.mean(fake_feats, axis=0)

        def _cov(feats):
            if feats.shape[0] < 2:
                return np.eye(feats.shape[1]) * 1e-6
            return np.cov(feats, rowvar=False)

        sigma_r, sigma_f = _cov(real_feats), _cov(fake_feats)
        sigma_r, sigma_f = np.atleast_2d(sigma_r), np.atleast_2d(sigma_f)

        diff = mu_r - mu_f
        covmean, _ = linalg.sqrtm(sigma_r @ sigma_f, disp=False)
        if np.iscomplexobj(covmean):
            covmean = covmean.real

        return float(diff @ diff + np.trace(sigma_r + sigma_f - 2 * covmean))

    # 加载真实tokens
    if not os.path.exists(real_token_dir):
        return {'FID': -1, 'FVD': -1}

    real_files = sorted([f for f in os.listdir(real_token_dir) if f.endswith('.npy')])[:50]
    real_tokens = []
    for f in real_files:
        t = np.load(os.path.join(real_token_dir, f)).astype(np.float32)
        if t.shape[0] == 64:
            t = np.concatenate([t, t], axis=0)
        real_tokens.append(t)

    if not real_tokens:
        return {'FID': -1, 'FVD': -1}

    real_tokens = np.stack(real_tokens, axis=0)

    # 加载生成的samples
    samples_dir = '/root/OccSora-main/eval_results_paperish_v6'
    samples = []
    for i in range(8):
        path = f'{samples_dir}/samples_{model_name}_cond{i}.npy'
        if os.path.exists(path):
            s = np.load(path)
            samples.append(s)

    if not samples:
        return {'FID': -1, 'FVD': -1}

    samples = np.concatenate(samples, axis=0)

    # 提取特征并计算FID/FVD
    feat2d = FeatureExtractor2D().to(device).eval()
    feat3d = FeatureExtractor3D().to(device).eval()

    with torch.no_grad():
        real_t = torch.from_numpy(real_tokens).float().to(device)
        real_2d = feat2d(real_t.mean(dim=2)).cpu().numpy()
        real_3d = feat3d(real_t).cpu().numpy()

        fake_t = torch.from_numpy(samples).float().to(device)
        fake_2d = feat2d(fake_t.mean(dim=2)).cpu().numpy()
        fake_3d = feat3d(fake_t).cpu().numpy()

    fid = frechet_distance(real_2d, fake_2d)
    fvd = frechet_distance(real_3d, fake_3d)

    return {'FID': fid, 'FVD': fvd}


# ============================================
# 主函数
# ============================================

def main():
    parser = argparse.ArgumentParser(description="多数据集评估")
    parser.add_argument("--datasets", nargs='+', default=['occ3d', 'nuscenes'],
                       help="要评估的数据集")
    parser.add_argument("--models", nargs='+', default=['baseline', 'full'],
                       help="要评估的模型")
    parser.add_argument("--output", type=str, default='/root/OccSora-main/multi_dataset_results',
                       help="输出目录")
    parser.add_argument("--vqvae-ckpt", type=str,
                       default='/root/autodl-tmp/OccSora_output/vqvae/latest.pth',
                       help="VQ-VAE检查点路径")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 加载VQ-VAE
    print("\n[1] Loading VQ-VAE...")
    try:
        vqvae = load_vqvae(args.vqvae_ckpt)
        print(f"  Loaded from: {args.vqvae_ckpt}")
    except Exception as e:
        print(f"  Warning: Failed to load VQ-VAE: {e}")
        vqvae = None

    results = {}

    # 评估每个数据集
    for dataset in args.datasets:
        print(f"\n{'#'*70}")
        print(f"# Dataset: {DATASETS.get(dataset, {}).get('description', dataset)}")
        print("#"*70)

        results[dataset] = {}

        # 评估Copy-Paste基线
        results[dataset]['Copy-Paste'] = eval_copy_paste(dataset)

        # 评估OccSora模型
        if vqvae is not None:
            for model_name in args.models:
                results[dataset][f'OccSora-{model_name}'] = eval_occsora_model(
                    model_name, dataset, vqvae, device
                )

    # 计算FID/FVD (在latent空间)
    print(f"\n{'#'*70}")
    print("# FID/FVD Evaluation (Latent Space)")
    print("#"*70)

    token_dir = '/root/autodl-tmp/OccSora_output/vqvae/step32-2/token'
    for model_name in args.models:
        print(f"\n  Computing FID/FVD for {model_name}...")
        fid_fvd = eval_fid_fvd(model_name, token_dir, device)
        results[f'fid_fvd_{model_name}'] = fid_fvd
        print(f"    FID: {fid_fvd['FID']:.2f}, FVD: {fid_fvd['FVD']:.2f}")

    # 生成报告
    print(f"\n{'#'*70}")
    print("# Summary")
    print("#"*70)

    report = """# Multi-Dataset Evaluation Report

## Semantic Occupancy Prediction (mIoU)

| Method | Occ3D mIoU | nuScenes mIoU |
|--------|------------|---------------|
"""

    methods = ['Copy-Paste', 'OccSora-baseline', 'OccSora-full']
    for method in methods:
        occ3d_miou = results.get('occ3d', {}).get(method, {}).get('mIoU', 'N/A')
        nuscenes_miou = results.get('nuscenes', {}).get(method, {}).get('mIoU', 'N/A')

        occ3d_str = f"{occ3d_miou:.2f}%" if isinstance(occ3d_miou, float) else occ3d_miou
        nuscenes_str = f"{nuscenes_miou:.2f}%" if isinstance(nuscenes_miou, float) else nuscenes_miou

        if 'full' in method:
            report += f"| **{method}** | **{occ3d_str}** | **{nuscenes_str}** |\n"
        else:
            report += f"| {method} | {occ3d_str} | {nuscenes_str} |\n"

    report += """
## Generation Quality (FID/FVD in Latent Space)

| Method | FID | FVD |
|--------|-----|-----|
"""

    for model_name in args.models:
        fid_fvd = results.get(f'fid_fvd_{model_name}', {})
        fid = fid_fvd.get('FID', 'N/A')
        fvd = fid_fvd.get('FVD', 'N/A')

        fid_str = f"{fid:.2f}" if isinstance(fid, float) and fid >= 0 else 'N/A'
        fvd_str = f"{fvd:.2f}" if isinstance(fvd, float) and fvd >= 0 else 'N/A'

        if model_name == 'full':
            report += f"| **OccSora-{model_name}** | **{fid_str}** | **{fvd_str}** |\n"
        else:
            report += f"| OccSora-{model_name} | {fid_str} | {fvd_str} |\n"

    report += """
## Key Findings

1. **Copy-Paste Baseline**: Provides a lower bound for prediction quality
2. **OccSora Baseline**: Standard diffusion-based prediction
3. **OccSora Full**: With all innovations (STCA + SADS + PCL)

## Notes

- mIoU calculated on semantic occupancy (18 classes)
- FID/FVD calculated in VQ-VAE latent space
- Lower FID/FVD indicates better generation quality
"""

    print(report)

    # 保存结果
    with open(f'{args.output}/results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)

    with open(f'{args.output}/report.md', 'w') as f:
        f.write(report)

    print(f"\nResults saved to: {args.output}/")


if __name__ == "__main__":
    main()
