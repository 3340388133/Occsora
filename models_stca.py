"""
DiT_STCA - 时空因果注意力增强的扩散Transformer
=================================================

架构级创新：将STCA集成到DiT模型架构中

与原DiT的区别：
1. 使用STCADiTBlock替换DiTBlock
2. 时间因果注意力确保t只能看到<=t（推理时也生效）
3. 空间窗口注意力利用3D局部相关性
4. 与原模型输入输出完全兼容

论文贡献：
- 模型架构级创新
- 训练和推理时都强制因果约束
- 生成更符合物理规律的时序占用
"""

import torch
import torch.nn as nn
import numpy as np
import math
from typing import Optional, Tuple

# 导入原DiT的组件
from timm.models.vision_transformer import Mlp

# 导入STCA组件
import sys
sys.path.insert(0, '/root/OccSora-main')
from Project.innovations.stca.stca_dit_block import STCADiTBlock


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """1D正弦余弦位置编码"""
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega

    out = np.einsum('m,d->md', pos, omega)
    emb_sin = np.sin(out)
    emb_cos = np.cos(out)
    emb = np.concatenate([emb_sin, emb_cos], axis=1)
    return emb


class TimestepEmbedder(nn.Module):
    """时间步嵌入"""
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class FinalLayer(nn.Module):
    """最终输出层"""
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class MyNet(nn.Module):
    """输入卷积网络"""
    def __init__(self, in_channels=128):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels=in_channels, out_channels=256, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(in_channels=256, out_channels=256, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class MLP_Trag(nn.Module):
    """轨迹MLP"""
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class DiT_STCA(nn.Module):
    """
    时空因果注意力增强的扩散Transformer

    架构创新：
    1. STCADiTBlock替换DiTBlock
    2. 时间因果注意力（训练+推理都生效）
    3. 空间窗口注意力（利用3D局部性）
    """

    def __init__(
        self,
        input_size=32,
        patch_size=2,
        in_channels=4,
        hidden_size=256,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        num_classes=1000,
        learn_sigma=True,
        num_frames=4,  # 时间帧数
        spatial_shape=(1, 25, 25),  # 单帧空间维度 (D, H, W)，默认按T=4帧、每帧25x25
        use_stca=True,  # 是否使用STCA（可切换对比）
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.use_stca = use_stca
        self.num_frames = num_frames
        self.spatial_shape = spatial_shape

        # 输入处理
        self.mlp_trag = MLP_Trag(input_size=64, hidden_size=128, output_size=256)
        self.my_net = MyNet()

        # 时间步嵌入
        self.t_embedder = TimestepEmbedder(hidden_size)

        # 位置嵌入
        num_patches = 2500
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches, hidden_size),
            requires_grad=True
        )

        # 核心创新：使用STCADiTBlock或原始DiTBlock
        if use_stca:
            self.blocks = nn.ModuleList([
                STCADiTBlock(
                    hidden_size,
                    num_heads,
                    mlp_ratio=mlp_ratio,
                    num_frames=num_frames,
                    spatial_shape=spatial_shape,
                ) for _ in range(depth)
            ])
            print(f"[DiT_STCA] Using STCADiTBlock x {depth}")
        else:
            # 回退到原始DiTBlock（用于对比实验）
            from models import DiTBlock
            self.blocks = nn.ModuleList([
                DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio)
                for _ in range(depth)
            ])
            print(f"[DiT_STCA] Using original DiTBlock x {depth}")

        # 输出层
        self.final_layer = FinalLayer(hidden_size, patch_size, 64)

        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # 位置编码初始化
        pos_dim = int(self.pos_embed.shape[-1])
        pos_embed = get_1d_sincos_pos_embed_from_grid(pos_dim, np.arange(1, 2501))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # 时间步嵌入初始化
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(self, x, t, y):
        """
        前向传播

        Args:
            x: (B, C, D, H, W) 输入
            t: (B,) 时间步
            y: (B, 64) 条件
        Returns:
            (B, 256, 4, 25, 25) 输出
        """
        batch_size = x.shape[0]
        y = y.reshape(batch_size, 64)
        y = self.mlp_trag(y)

        x1 = self.my_net(x)
        x1 = x1.reshape(batch_size, 256, 2500)
        x1 = x1.permute(0, 2, 1)

        x2 = self.pos_embed
        x = x1 + x2

        t = self.t_embedder(t)
        c = t + y

        # 通过STCA块
        for block in self.blocks:
            x = block(x, c)

        x = self.final_layer(x, c)
        x = x.permute(0, 2, 1)
        x = x.reshape(batch_size, 256, 4, 25, 25)

        return x

    def forward_with_cfg(self, x, t, y, cfg_scale):
        """支持CFG的前向传播"""
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, y)
        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)


# =============================================================================
# 模型配置
# =============================================================================

def DiT_STCA_XL_2(**kwargs):
    return DiT_STCA(depth=28, hidden_size=256, patch_size=2, num_heads=16, **kwargs)

def DiT_STCA_L_2(**kwargs):
    return DiT_STCA(depth=24, hidden_size=256, patch_size=2, num_heads=16, **kwargs)

def DiT_STCA_B_2(**kwargs):
    return DiT_STCA(depth=12, hidden_size=256, patch_size=2, num_heads=8, **kwargs)


# 模型注册表
DiT_STCA_models = {
    'DiT-STCA-XL/2': DiT_STCA_XL_2,
    'DiT-STCA-L/2': DiT_STCA_L_2,
    'DiT-STCA-B/2': DiT_STCA_B_2,
}
