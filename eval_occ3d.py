#!/usr/bin/env python3
"""
Occ3D格式评估脚本
评估4D占用预测的mIoU指标
"""

import os
import numpy as np
import torch
from tqdm import tqdm
import json
from collections import defaultdict

# Occ3D-nuScenes类别定义 (17类)
OCC3D_CLASSES = {
    0: 'empty',
    1: 'barrier',
    2: 'bicycle',
    3: 'bus',
    4: 'car',
    5: 'construction_vehicle',
    6: 'motorcycle',
    7: 'pedestrian',
    8: 'traffic_cone',
    9: 'trailer',
    10: 'truck',
    11: 'driveable_surface',
    12: 'other_flat',
    13: 'sidewalk',
    14: 'terrain',
    15: 'manmade',
    16: 'vegetation',
    17: 'free'
}

class Occ3DEvaluator:
    """Occ3D格式的mIoU评估器"""

    def __init__(self, num_classes=18, ignore_index=0):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.reset()

    def reset(self):
        self.confusion_matrix = np.zeros(
            (self.num_classes, self.num_classes), dtype=np.int64
        )

    def update(self, pred, gt):
        """更新混淆矩阵"""
        mask = (gt != self.ignore_index)
        pred = pred[mask]
        gt = gt[mask]

        # 限制范围
        pred = np.clip(pred, 0, self.num_classes - 1)
        gt = np.clip(gt, 0, self.num_classes - 1)

        # 更新混淆矩阵
        for p, g in zip(pred.flatten(), gt.flatten()):
            self.confusion_matrix[g, p] += 1

    def compute_miou(self):
        """计算mIoU"""
        # IoU = TP / (TP + FP + FN)
        tp = np.diag(self.confusion_matrix)
        fp = self.confusion_matrix.sum(axis=0) - tp
        fn = self.confusion_matrix.sum(axis=1) - tp

        iou = tp / (tp + fp + fn + 1e-10)

        # 排除empty类(index 0)
        valid_classes = [i for i in range(1, self.num_classes)
                        if self.confusion_matrix[i].sum() > 0]

        if len(valid_classes) == 0:
            return 0.0, {}

        miou = np.mean([iou[i] for i in valid_classes])

        per_class_iou = {
            OCC3D_CLASSES.get(i, f'class_{i}'): iou[i] * 100
            for i in valid_classes
        }

        return miou * 100, per_class_iou


def load_gt_data(gt_dir, scene_id, sample_token):
    """加载GT占用数据"""
    gt_path = os.path.join(gt_dir, f"scene-{scene_id:04d}", sample_token, "labels.npz")
    if os.path.exists(gt_path):
        data = np.load(gt_path)
        return data['semantics']  # (200, 200, 16)
    return None


def get_all_samples(gt_dir):
    """获取所有样本"""
    samples = []
    scenes = sorted([d for d in os.listdir(gt_dir) if d.startswith('scene-')])

    for scene in scenes:
        scene_path = os.path.join(gt_dir, scene)
        if os.path.isdir(scene_path):
            tokens = sorted(os.listdir(scene_path))
            for token in tokens:
                if os.path.isdir(os.path.join(scene_path, token)):
                    scene_id = int(scene.split('-')[1])
                    samples.append((scene_id, token))

    return samples


def evaluate_copy_paste(gt_dir, num_samples=1000):
    """评估Copy&Paste基线 - 用当前帧预测下一帧"""
    print("="*60)
    print("Evaluating Copy&Paste baseline (predict next frame)")
    print("="*60)

    evaluator = Occ3DEvaluator(num_classes=18, ignore_index=0)
    samples = get_all_samples(gt_dir)[:num_samples]

    print(f"Total samples: {len(samples)}")

    # 按场景分组
    scene_samples = {}
    for scene_id, token in samples:
        if scene_id not in scene_samples:
            scene_samples[scene_id] = []
        scene_samples[scene_id].append(token)

    count = 0
    for scene_id, tokens in tqdm(scene_samples.items()):
        tokens = sorted(tokens)
        for i in range(len(tokens) - 1):
            # 用当前帧预测下一帧
            curr_gt = load_gt_data(gt_dir, scene_id, tokens[i])
            next_gt = load_gt_data(gt_dir, scene_id, tokens[i+1])
            if curr_gt is None or next_gt is None:
                continue
            # Copy&Paste: 用当前帧作为下一帧的预测
            pred = curr_gt.copy()
            evaluator.update(pred.flatten(), next_gt.flatten())
            count += 1
            if count >= num_samples:
                break
        if count >= num_samples:
            break

    miou, per_class = evaluator.compute_miou()
    print(f"\nCopy&Paste (next frame) mIoU: {miou:.2f}%")
    return miou, per_class


def evaluate_random(gt_dir, num_samples=1000):
    """评估Random基线"""
    print("="*60)
    print("Evaluating Random baseline (Occ3D format)")
    print("="*60)

    evaluator = Occ3DEvaluator(num_classes=18, ignore_index=0)
    samples = get_all_samples(gt_dir)[:num_samples]

    for i, (scene_id, token) in enumerate(tqdm(samples)):
        gt = load_gt_data(gt_dir, scene_id, token)
        if gt is None:
            continue
        # Random: 随机预测
        pred = np.random.randint(0, 18, gt.shape)
        evaluator.update(pred.flatten(), gt.flatten())

    miou, per_class = evaluator.compute_miou()
    print(f"\nRandom mIoU: {miou:.2f}%")
    return miou, per_class


def evaluate_majority_class(gt_dir, num_samples=1000):
    """评估Majority Class基线"""
    print("="*60)
    print("Evaluating Majority Class baseline")
    print("="*60)

    evaluator = Occ3DEvaluator(num_classes=18, ignore_index=0)
    samples = get_all_samples(gt_dir)[:num_samples]

    for i, (scene_id, token) in enumerate(tqdm(samples)):
        gt = load_gt_data(gt_dir, scene_id, token)
        if gt is None:
            continue
        pred = np.zeros_like(gt)
        pred.fill(11)  # driveable_surface
        evaluator.update(pred.flatten(), gt.flatten())

    miou, per_class = evaluator.compute_miou()
    print(f"\nMajority Class mIoU: {miou:.2f}%")
    return miou, per_class


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt-dir", default="/root/autodl-tmp/gts/gts")
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--output", default="/root/OccSora-main/occ3d_results.json")
    args = parser.parse_args()

    results = {}

    # 评估各基线
    miou, per_class = evaluate_copy_paste(args.gt_dir, args.num_samples)
    results['copy_paste'] = {'miou': miou, 'per_class': per_class}

    miou, per_class = evaluate_random(args.gt_dir, args.num_samples)
    results['random'] = {'miou': miou, 'per_class': per_class}

    miou, per_class = evaluate_majority_class(args.gt_dir, args.num_samples)
    results['majority'] = {'miou': miou, 'per_class': per_class}

    # 保存结果
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}")

    # 打印汇总
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"{'Method':<20} {'mIoU':>10}")
    print("-"*30)
    for name, res in results.items():
        print(f"{name:<20} {res['miou']:>10.2f}%")


if __name__ == "__main__":
    main()
