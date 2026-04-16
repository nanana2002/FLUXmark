#!/usr/bin/env python3
import os, json
merge_path = os.path.join('result', 'merge.json')
with open(merge_path) as f:
    raw_data = json.load(f)
# 支持两种结构：扁平结构 或嵌套在 overall 下
if 'overall' in raw_data and isinstance(raw_data['overall'], dict):
    merge_data = raw_data['overall']
else:
    merge_data = raw_data
print(f"读取到 {len(merge_data)} 个攻击类型")
table1_entries = []
for name, metrics in merge_data.items():
    if not isinstance(metrics, dict):
        continue
    table1_entries.append({
        'name': name,
        'mean_cosine': metrics.get('mean_cosine', 0),
        'retention': metrics.get('mean_retention_rate', 0) * 100,
        'auc': metrics.get('auc', 0),
        'tpr': metrics.get('tpr_at_0.1_fpr', 0),
    })
table2_entries = []
for name, metrics in merge_data.items():
    if not isinstance(metrics, dict):
        continue
    if 'mean_iou' in metrics or 'mean_patch_auc' in metrics:
        table2_entries.append({
            'name': name,
            'patch_auc': metrics.get('mean_patch_auc', 0),
            'f1': metrics.get('mean_f1', 0),
            'iou': metrics.get('mean_iou', 0),
        })
print(f"Table 1: {len(table1_entries)} 条")
print(f"Table 2: {len(table2_entries)} 条")
def fmt(name):
    mapping = {
        'jpeg_75': 'JPEG Q=75', 'jpeg_50': 'JPEG Q=50', 'jpeg_30': 'JPEG Q=30',
        'blur_0.5': 'Gaussian Blur ($\\\\sigma=0.5$)',
        'blur_1.0': 'Gaussian Blur ($\\\\sigma=1.0$)',
        'blur_2.0': 'Gaussian Blur ($\\\\sigma=2.0$)',
        'crop_0.75': 'Center Crop (75\\\\%)', 'crop_0.50': 'Center Crop (50\\\\%)',
        'noise_0.03': 'Gaussian Noise ($\\\\sigma=0.03$)',
        'noise_0.05': 'Gaussian Noise ($\\\\sigma=0.05$)',
        'noise_0.10': 'Gaussian Noise ($\\\\sigma=0.10$)',
        'resize_0.75': 'Resize (75\\\\%)', 'resize_0.50': 'Resize (50\\\\%)',
        'brightness_0.8': 'Brightness 0.8', 'brightness_1.2': 'Brightness 1.2',
        'contrast_0.8': 'Contrast 0.8', 'contrast_1.2': 'Contrast 1.2',
        'black_block_center': 'Black Block (Center)',
        'black_block_random': 'Black Block (Random)',
        'sdedit_0.3': 'SDEdit ($\\\\sigma=0.3$)',
        'sdxl_style_oil_painting': 'SDXL: Oil Painting',
        'sdxl_style_sketch': 'SDXL: Sketch',
        'sdxl_style_watercolor': 'SDXL: Watercolor',
        'sdxl_style_cyberpunk': 'SDXL: Cyberpunk',
        'sdxl_style_anime': 'SDXL: Anime',
        'fluxfill_center': 'FluxFill (Center)',
        'fluxfill_random': 'FluxFill (Random)',
    }
    return mapping.get(name, name.replace('_', ' ').title())
def order_key(x):
    name = x['name']
    for i, prefix in enumerate([
        'jpeg_', 'blur_', 'crop_', 'noise_', 'resize_',
        'brightness_', 'contrast_', 'black_block_',
        'sdedit_', 'sdxl_', 'fluxfill_'
    ]):
        if name.startswith(prefix):
            return (i, name)
    return (99, name)
def bold(val, th):
    return f"\\\\textbf{{{val:.4f}}}" if val >= th else f"{val:.4f}"
def bold_pct(val, th):
    s = f"{val:.2f}\\\\%"
    return f"\\\\textbf{{{s}}}" if val >= th else s
table1 = r"""\begin{table}[t]
\centering
\caption{Global Robustness Evaluation (Copyright Verification)}
\label{tab:robustness}
\resizebox{0.98\columnwidth}{!}{%
\begin{tabular}{lcccc}
\toprule
\textbf{Attack Type} & \textbf{Mean Cosine} & \textbf{Retention} & \textbf{AUC} & \textbf{TPR@0.1\%FPR} \\
\midrule
""" + "\n".join([
    f"{fmt(e['name'])} & {e['mean_cosine']:.4f} & {bold_pct(e['retention'], 80.0)} & {bold(e['auc'], 0.95)} & {bold(e['tpr'], 0.95)} \\\\"
    for e in sorted(table1_entries, key=order_key)
]) + r"""
\bottomrule
\end{tabular}%
}
\end{table}
"""
table2 = ""
if table2_entries:
    table2 = r"""\begin{table}[t]
\centering
\caption{Tamper Localization Evaluation (Fragility)}
\label{tab:tamper}
\resizebox{0.85\columnwidth}{!}{%
\begin{tabular}{lccc}
\toprule
\textbf{Attack Type} & \textbf{Patch-AUC} & \textbf{F1-Score} & \textbf{IoU} \\
\midrule
""" + "\n".join([
    f"{fmt(e['name'])} & {bold(e['patch_auc'], 0.90)} & {bold(e['f1'], 0.50)} & {bold(e['iou'], 0.40)} \\\\"
    for e in sorted(table2_entries, key=order_key)
]) + r"""
\bottomrule
\end{tabular}%
}
\end{table}
"""
os.makedirs('result/latex', exist_ok=True)
with open('result/latex/table1_robustness.tex', 'w') as f: f.write(table1)
with open('result/latex/table2_tamper.tex', 'w') as f: f.write(table2)
with open('result/latex/tables_combined.tex', 'w') as f:
    f.write(table1 + "\n\n" + table2)
print("\n✅ 生成完成")
print("   result/latex/table1_robustness.tex")
print("   result/latex/table2_tamper.tex")
print("   result/latex/tables_combined.tex")




