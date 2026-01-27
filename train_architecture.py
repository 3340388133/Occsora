"""
训练脚本 - 架构级创新版本 (patched)
==========================

使用架构级创新训练OccSora模型：
1. DiT_STCA: 时空因果注意力集成到模型架构
2. SADS: 场景自适应扩散（训练时也使用）
3. PCL: 物理一致性损失

用法:
    # 使用STCA架构
    python train_architecture.py --model DiT-STCA-XL/2 --use-stca

    # 对比实验（不使用STCA）
    python train_architecture.py --model DiT-STCA-XL/2 --no-stca
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import shutil
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import time

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

sys.path.insert(0, '/root/OccSora-main')

# 导入架构级创新模型
from models_stca import DiT_STCA_models

# 导入扩散过程
from diffusion import create_diffusion

# 导入创新模块
from Project.innovations.sads import (
    SceneComplexityEncoder,
    AdaptiveNoiseScheduler,
)
from Project.innovations.pcl import PhysicsConsistentLoss


class OccupancyDataset(Dataset):
    """占用数据集 - 加载真实token/cond并支持数据增强"""

    def __init__(
        self,
        data_dir: str,
        split: str = 'train',
        num_samples: int = 1000,
        augment_factor: int = 1,
        max_samples: int = 0,
    ):
        self.data_dir = data_dir
        self.token_dir = os.path.join(data_dir, 'vqvae/step32-2/token')
        self.cond_dir = os.path.join(data_dir, 'vqvae/step32-2/gt_mode')

        self.num_samples = int(num_samples)
        self.augment_factor = int(augment_factor)
        self.max_samples = int(max_samples)

        self.base_samples = self._load_samples(split)

        total = len(self.base_samples) * max(self.augment_factor, 1)
        if self.max_samples > 0:
            total = min(total, self.max_samples)
        self._total_len = int(total)

        print(
            f"[Dataset] base={len(self.base_samples)} augment_factor={self.augment_factor} "
            f"max_samples={self.max_samples} effective_len={len(self)}"
        )

    def _load_samples(self, split):
        if os.path.exists(self.token_dir):
            files = [f for f in os.listdir(self.token_dir) if f.endswith('.npy')]
            return sorted(files)

        print(f"[Dataset] Warning: {self.token_dir} not found")
        return list(range(self.num_samples))

    def __len__(self):
        return self._total_len

    def __getitem__(self, idx):
        base_len = len(self.base_samples)
        if base_len <= 0:
            x_start = torch.randn(128, 4, 25, 25)
            condition = torch.randn(64)
            return x_start, condition

        base_idx = idx % base_len
        aug_idx = idx // base_len

        if isinstance(self.base_samples[base_idx], str):
            token_path = os.path.join(self.token_dir, self.base_samples[base_idx])
            cond_path = os.path.join(self.cond_dir, self.base_samples[base_idx])

            token = np.load(token_path)  # (64, 4, 25, 25)
            cond = np.load(cond_path)  # (31,) typically

            if aug_idx > 0:
                noise_scale = 0.1 * (aug_idx / max(self.augment_factor, 1))
                token = token + np.random.randn(*token.shape).astype(np.float32) * noise_scale
                if aug_idx % 2 == 1:
                    token = np.flip(token, axis=-1).copy()
                if aug_idx % 3 == 0:
                    token = np.flip(token, axis=-2).copy()

            x_start = np.concatenate([token, token], axis=0)  # (128, 4, 25, 25)

            if len(cond) < 64:
                cond = np.pad(cond, (0, 64 - len(cond)), mode='constant')
            else:
                cond = cond[:64]

            return torch.from_numpy(x_start).float(), torch.from_numpy(cond).float()

        x_start = torch.randn(128, 4, 25, 25)
        condition = torch.randn(64)
        return x_start, condition


def train_epoch(
    model,
    dataloader,
    optimizer,
    diffusion,
    device,
    args,
    scaler=None,
    complexity_encoder=None,
    noise_scheduler=None,
    pcl_loss_fn=None,
):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    pbar = tqdm(dataloader, desc="Training")

    for batch_idx, (x_start, conditions) in enumerate(pbar):
        x_start = x_start.to(device)
        conditions = conditions.to(device)

        optimizer.zero_grad(set_to_none=True)

        # 随机时间步
        t = torch.randint(0, diffusion.num_timesteps, (x_start.size(0),), device=device)

        # 生成噪声
        if args.use_sads and complexity_encoder is not None:
            with torch.no_grad():
                complexity = complexity_encoder(conditions)
            noise_weights = 1.0 + 0.2 * complexity.mean(dim=1, keepdim=True)
            noise = torch.randn_like(x_start) * noise_weights.view(-1, 1, 1, 1, 1)
        else:
            noise = torch.randn_like(x_start)

        # 加噪
        x_t = diffusion.q_sample(x_start, t, noise=noise)

        autocast_enabled = (device.type == "cuda")
        with torch.cuda.amp.autocast(enabled=autocast_enabled):
            # 模型预测
            model_output = model(x_t, t, conditions)

            # 基础损失
            if model.learn_sigma:
                noise_pred, _ = model_output.chunk(2, dim=1)
            else:
                noise_pred = model_output

            mse_loss = nn.functional.mse_loss(noise_pred.float(), noise.float())

            # 物理一致性损失
            if args.use_pcl and pcl_loss_fn is not None:
                pred_x0 = diffusion._predict_xstart_from_eps(x_t=x_t, t=t, eps=noise_pred)
                pcl_losses = pcl_loss_fn(pred_x0)
                physics_loss = pcl_losses['total']
                loss = mse_loss + args.pcl_weight * physics_loss
            else:
                loss = mse_loss

        if scaler is not None and scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / len(dataloader)


def main():
    parser = argparse.ArgumentParser(description="OccSora Architecture-Level Training")

    # 模型参数
    parser.add_argument("--model", type=str, default="DiT-STCA-XL/2",
                        choices=list(DiT_STCA_models.keys()))

    # 架构级创新开关
    parser.add_argument("--use-stca", action="store_true", default=False,
                        help="使用STCA架构")
    parser.add_argument("--no-stca", action="store_true",
                        help="不使用STCA（用于对比实验）")
    parser.add_argument("--use-sads", action="store_true", default=False,
                        help="使用SADS场景自适应调度")
    parser.add_argument("--use-pcl", action="store_true", default=False,
                        help="使用PCL物理一致性损失")
    parser.add_argument("--pcl-weight", type=float, default=0.1,
                        help="PCL损失权重")

    # 训练参数
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--data-dir", type=str, default="/root/autodl-tmp/OccSora_output")
    parser.add_argument("--num-samples", type=int, default=1000,
                        help="(模拟数据) base样本数；总样本数=base×augment_factor（再受 --max-samples 限制）")
    parser.add_argument("--augment-factor", type=int, default=1,
                        help="(数据增强) 每个base样本扩增次数；总样本数=base×augment_factor（再受 --max-samples 限制）")
    parser.add_argument("--max-samples", type=int, default=0,
                        help="(可选) 限制训练集总样本数上限；例如 500 表示只用前500条(按增强展开后的顺序)")
    parser.add_argument("--output-dir", type=str, default="/root/autodl-tmp/model")
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--save-optimizer", action="store_true", default=False,
                        help="保存optimizer状态（文件更大；默认关闭以节省磁盘）")

    args = parser.parse_args()

    # 处理STCA开关
    use_stca = args.use_stca and not args.no_stca

    print("=" * 60)
    print("OccSora Architecture-Level Training")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"STCA (Architecture-Level): {use_stca}")
    print(f"SADS (Scene-Adaptive): {args.use_sads}")
    print(f"PCL (Physics-Consistent): {args.use_pcl}")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 创建模型
    model = DiT_STCA_models[args.model](use_stca=use_stca).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # 创建扩散过程
    diffusion = create_diffusion(timestep_respacing="")

    # 创新模块
    complexity_encoder = None
    noise_scheduler = None
    pcl_loss_fn = None

    if args.use_sads:
        complexity_encoder = SceneComplexityEncoder(input_dim=64).to(device)
        noise_scheduler = AdaptiveNoiseScheduler().to(device)

    if args.use_pcl:
        pcl_loss_fn = PhysicsConsistentLoss().to(device)

    # 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    # 数据集
    dataset = OccupancyDataset(
        args.data_dir,
        split='train',
        num_samples=args.num_samples,
        augment_factor=args.augment_factor,
        max_samples=args.max_samples,
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    # 训练循环
    os.makedirs(args.output_dir, exist_ok=True)

    def _atomic_torch_save(obj, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp_path = path + ".tmp"
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass

        # First try default (zip) format, then fall back to legacy format.
        try:
            torch.save(obj, tmp_path)
            os.replace(tmp_path, path)
            return
        except Exception:
            # Clean up tmp before retry
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except OSError:
                pass

        torch.save(obj, tmp_path, _use_new_zipfile_serialization=False)
        os.replace(tmp_path, path)

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")

        avg_loss = train_epoch(
            model, dataloader, optimizer, diffusion, device, args,
            scaler, complexity_encoder, noise_scheduler, pcl_loss_fn
        )

        print(f"Average Loss: {avg_loss:.4f}")

        # 保存检查点
        if ((epoch + 1) % args.save_every == 0) or (epoch + 1 == args.epochs):
            ckpt_path = os.path.join(args.output_dir, f"dit_stca_epoch{epoch+1}.pt")

            ckpt = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'args': args,
            }
            if args.save_optimizer:
                ckpt['optimizer_state_dict'] = optimizer.state_dict()

            try:
                _atomic_torch_save(ckpt, ckpt_path)
                print(f"Saved checkpoint: {ckpt_path}")
            except Exception as e:
                total, used, free = shutil.disk_usage(os.path.dirname(ckpt_path) or ".")
                print(f"[ERROR] Failed to save checkpoint: {ckpt_path}")
                print(f"[ERROR] {type(e).__name__}: {e}")
                print(f"[HINT] Disk free: {free/1024/1024/1024:.2f} GiB (filesystem may be full).")
                raise


if __name__ == "__main__":
    main()
