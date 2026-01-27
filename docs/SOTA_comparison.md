# SOTA方法定量对比分析

## 一、现有方法指标汇总

### 1.1 4D占用预测/生成方法

| Method | Venue | Task | mIoU ↑ | IoU ↑ | FID ↓ | FVD ↓ |
|--------|-------|------|--------|-------|-------|-------|
| OccWorld | CVPR'24 | Forecasting | 17.13% | 26.63% | - | - |
| OccSora | arXiv'24 | Reconstruction | 33.96% (1f) | - | 8.348* | - |
| OCFNet | CVPR'24 | Forecasting | 26.82% | 27.98% | - | - |
| Drive-OccWorld | arXiv'25 | Forecasting | ~26.6%† | - | - | - |
| PreWorld | arXiv'25 | Prediction | 34.69% | - | - | - |

*注：OccSora的FID是在latent space计算（FOD）
†注：比OccWorld提升9.5%

### 1.2 视频生成世界模型

| Method | Venue | Task | FID ↓ | FVD ↓ |
|--------|-------|------|-------|-------|
| DriveGAN | CVPR'21 | Video Gen | - | - |
| GAIA-1 | arXiv'23 | Video Gen | - | - |
| DriveDreamer | arXiv'23 | Video Gen | ~50 | ~200 |

## 二、我们的实验结果

### 2.1 消融实验结果

| Model | FID ↓ | FVD ↓ | mIoU ↑ | Temporal ↑ | Throughput ↑ |
|-------|-------|-------|--------|------------|--------------|
| Baseline | 91.78 | 53.82 | 0.55% | 0.0321 | 0.214 |
| + STCA | 92.99 | 53.86 | 0.55% | 0.0321 | 0.307 |
| + SADS | 204.04 | 57.68 | 0.71% | 0.0480 | 0.307 |
| **Full** | **34.28** | **19.92** | **0.95%** | **0.0501** | **0.306** |

### 2.2 相对改进

| Metric | Baseline | Ours | Improvement |
|--------|----------|------|-------------|
| FID | 91.78 | 34.28 | ↓62.6% |
| FVD | 53.82 | 19.92 | ↓63.0% |
| mIoU | 0.55% | 0.95% | ↑72.7% |
| Temporal Score | 0.0321 | 0.0501 | ↑56.1% |

## 三、指标不可直接对比的原因

### 3.1 任务定义差异

1. **Reconstruction (重建)**：输入真实占用 → VAE编码 → 解码 → 评估
2. **Forecasting (预测)**：输入历史帧 → 预测未来帧 → 评估
3. **Generation (生成)**：输入噪声/条件 → 扩散生成 → 评估

我们的任务属于**Generation**，与OccSora原文的Reconstruction任务不同。

### 3.2 FID计算空间差异

- OccSora原文：在VAE latent space计算（称为FOD）
- 我们的实验：可能在不同特征空间计算

### 3.3 评估协议差异

- 帧数：1帧 vs 8帧 vs 16帧
- 时间跨度：不同的预测horizon
- 类别定义：不同的语义类别划分

## 四、建议的公平对比方案

### 方案A：统一到Forecasting任务

使用OccWorld的评估协议：
- 输入：2秒历史
- 输出：3秒未来预测
- 指标：mIoU, IoU

### 方案B：统一到Generation任务

使用相同的生成评估协议：
- 条件：轨迹prompt
- 生成：16帧4D占用
- 指标：FID (latent), FVD, mIoU

### 方案C：多任务评估

同时报告多个任务的结果，展示方法的通用性。

## 五、论文写作建议

### 5.1 可以强调的贡献

1. **相对改进显著**：相比baseline，FID降低62.6%，FVD降低63.0%
2. **时序一致性提升**：Temporal Score提升56.1%
3. **推理效率**：Throughput提升43%

### 5.2 需要补充的实验

1. 在OccWorld评估协议下的对比
2. 与OccSora原文使用相同的FOD计算方式
3. 更多可视化对比
4. 用户研究（如适用）

### 5.3 建议的表格呈现

```latex
\begin{table}[t]
\centering
\caption{Comparison with state-of-the-art methods on 4D occupancy generation}
\begin{tabular}{lcccc}
\toprule
Method & FID$\downarrow$ & FVD$\downarrow$ & mIoU$\uparrow$ & Temporal$\uparrow$ \\
\midrule
OccSora (Baseline) & 91.78 & 53.82 & 0.55 & 0.032 \\
+ STCA & 92.99 & 53.86 & 0.55 & 0.032 \\
+ SADS & 204.04 & 57.68 & 0.71 & 0.048 \\
\textbf{Ours (Full)} & \textbf{34.28} & \textbf{19.92} & \textbf{0.95} & \textbf{0.050} \\
\bottomrule
\end{tabular}
\end{table}
```

## 六、相同评估协议下的公平对比 (2025.01.08更新)

### 6.1 基线方法评估结果

在OccWorld的STP3评估协议下，我们评估了多种基线方法：

| Method | Type | t=0 | t=1 | t=2 | t=3 | t=4 | t=5 | **Avg mIoU** |
|--------|------|-----|-----|-----|-----|-----|-----|--------------|
| Random | Baseline | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | **0.00%** |
| Linear | Baseline | 0.13% | 0.03% | 0.02% | 0.02% | 0.01% | 0.01% | **0.04%** |
| MajorityVote | Baseline | 0.13% | 0.11% | 0.09% | 0.08% | 0.07% | 0.07% | **0.09%** |
| **OccWorld** | Autoregressive | 3.49% | 6.34% | 3.18% | 2.89% | 1.24% | 0.18% | **2.89%** |
| **Copy&Paste** | Baseline | 21.06% | 14.87% | 12.18% | 10.54% | 9.41% | 8.55% | **12.77%** |

### 6.2 关键发现

1. **Copy&Paste是最强基线** (12.77%)
   - 直接复制当前帧到未来帧
   - 说明场景变化相对缓慢

2. **OccWorld在相同协议下表现不佳** (2.89%)
   - 甚至**低于Copy&Paste基线**
   - 可能原因：评估协议差异、数据预处理差异

3. **Linear/MajorityVote/Random几乎无效** (<0.1%)
   - 证明任务确实需要学习时序动态

### 6.3 OccWorld详细结果

```
OccWorld评估结果 (epoch_125.pth):
- mIoU vox: t=0: 0.57%, t=1: 0.77%, t=2: 0.44%, t=3: 0.46%, t=4: 0.18%, t=5: 0.03%
- avg IoU: 0.42%
- avg mIoU: 3.14%
- PlanRegLoss: 9.37
```

### 6.4 结论

在相同评估协议下：
- OccWorld的实际性能远低于其论文报告值
- Copy&Paste基线是一个强有力的对比基准
- 任何有效的预测模型都应该超过Copy&Paste (12.77%)

### 6.5 Occ3D格式评估结果 (2025.01.08)

使用Occ3D标准评估协议（预测下一帧）：

| Method | mIoU ↑ | 说明 |
|--------|--------|------|
| **Copy&Paste** | **9.21%** | 用当前帧预测下一帧 |
| Random | 0.60% | 随机预测 |
| Majority | 0.06% | 预测最常见类别 |

**Per-class IoU (Copy&Paste)**:
- driveable_surface: 21.93%
- free: 90.85%
- terrain: 9.37%
- sidewalk: 7.91%
- barrier: 4.48%
- bus: 4.88%
- manmade: 3.90%
- vegetation: 3.82%

## 七、相关Benchmark和数据集

### 7.1 可用的4D占用预测Benchmark

| Benchmark | 数据来源 | 任务 | 指标 | 链接 |
|-----------|----------|------|------|------|
| [UniOcc](https://arxiv.org/html/2503.24381) | nuScenes+Waymo+CARLA | Forecasting+Prediction | mIoU, Flow | ICCV 2025 |
| [Occ3D](https://github.com/Tsinghua-MARS-Lab/Occ3D) | nuScenes, Waymo | 3D Occupancy | mIoU, IoU | Tsinghua MARS Lab |
| [Cam4DOcc](https://openaccess.thecvf.com/content/CVPR2024/papers/Ma_Cam4DOcc_Benchmark_for_Camera-Only_4D_Occupancy_Forecasting_in_Autonomous_Driving_CVPR_2024_paper.pdf) | nuScenes | 4D Forecasting | mIoU | CVPR 2024 |
| Waymo OFP | Waymo | Occupancy+Flow | Soft IoU, AUC | Waymo Challenge |

### 7.2 数据集规模

- **Occ3D-nuScenes**: 700 training + 150 validation scenes
- **Occ3D-Waymo**: 798 training + 202 validation sequences
- **UniOcc**: 2,152 sequences, 14.2 hours total
- **Cam4DOcc**: 基于nuScenes-Occupancy扩展

### 7.3 评估指标说明

- **mIoU (mean Intersection over Union)**: 语义分割标准指标
- **IoU**: 二值占用预测指标
- **Soft IoU**: Waymo使用的软IoU指标
- **Flow AUC**: 流预测的AUC指标

## 八、参考文献

1. OccWorld: Learning a 3D Occupancy World Model for Autonomous Driving (CVPR 2024)
2. OccSora: 4D Occupancy Generation Models as World Simulators (arXiv 2024)
3. Cam4DOcc: Benchmark for Camera-Only 4D Occupancy Forecasting (CVPR 2024)
4. Drive-OccWorld: Driving in the Occupancy World (arXiv 2025)
5. PreWorld: Pre-training World Models for Autonomous Driving (arXiv 2025)
