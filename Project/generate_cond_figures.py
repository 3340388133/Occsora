#!/usr/bin/env python3
"""生成按condition分组的折线图和表格"""

import json
import numpy as np
import matplotlib.pyplot as plt
import os

# 加载数据
with open('/root/OccSora-main/eval_results_paperish_v6/metrics_by_cond.json', 'r') as f:
    data = json.load(f)

output_dir = '/root/OccSora-main/Project/demo_output'
os.makedirs(output_dir, exist_ok=True)

models = ['baseline', 'stca', 'sads', 'full']
model_labels = ['Baseline', '+STCA', '+SADS', 'Full']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
conds = list(range(8))

# 提取指标
metrics_to_plot = {
    'miou': ('mIoU (%)', 100),
    'fid': ('FID↓', 1),
    'fvd': ('FVD↓', 1),
    'physics_score': ('Physics Score (%)', 100),
    'flicker_index': ('Flicker Index↓', 1),
    'diversity': ('Diversity', 1),
}

# 生成折线图
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for idx, (metric, (label, scale)) in enumerate(metrics_to_plot.items()):
    ax = axes[idx]
    for m_idx, model in enumerate(models):
        values = [data[model][str(c)][metric] * scale for c in conds]
        ax.plot(conds, values, marker='o', label=model_labels[m_idx], color=colors[m_idx], linewidth=2)
    ax.set_xlabel('Condition Index')
    ax.set_ylabel(label)
    ax.set_title(f'{label} by Condition')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{output_dir}/metrics_by_cond.png', dpi=150, bbox_inches='tight')
print(f"Saved: {output_dir}/metrics_by_cond.png")

# 生成表格 (Markdown)
table_md = """# Metrics by Condition

## mIoU (%) by Condition

| Cond | Baseline | +STCA | +SADS | Full |
|------|----------|-------|-------|------|
"""

for c in conds:
    row = f"| {c} |"
    for model in models:
        val = data[model][str(c)]['miou'] * 100
        row += f" {val:.3f} |"
    table_md += row + "\n"

table_md += """
## FID by Condition

| Cond | Baseline | +STCA | +SADS | Full |
|------|----------|-------|-------|------|
"""

for c in conds:
    row = f"| {c} |"
    for model in models:
        val = data[model][str(c)]['fid']
        row += f" {val:.2f} |"
    table_md += row + "\n"

table_md += """
## Physics Score (%) by Condition

| Cond | Baseline | +STCA | +SADS | Full |
|------|----------|-------|-------|------|
"""

for c in conds:
    row = f"| {c} |"
    for model in models:
        val = data[model][str(c)]['physics_score'] * 100
        row += f" {val:.2f} |"
    table_md += row + "\n"

with open(f'{output_dir}/metrics_by_cond.md', 'w') as f:
    f.write(table_md)
print(f"Saved: {output_dir}/metrics_by_cond.md")

print("\nDone!")
