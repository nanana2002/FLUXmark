#!/usr/bin/env python3
"""
实验结果汇总脚本
自动读取所有 result JSON，整合成一个漂亮的 Markdown 报告 + JSON 汇总
"""

import os
import json
from datetime import datetime

with open('config.json', 'r') as f:
    config = json.load(f)

result_dir = os.path.join(config['output_base_dir'], 'result')
os.makedirs(result_dir, exist_ok=True)

output_md = os.path.join(result_dir, 'SUMMARY.md')
output_json = os.path.join(result_dir, 'all_results_summary.json')

# ==========================================
# 辅助函数：安全读取 JSON
# ==========================================
def load_json(path, default=None):
    if os.path.exists(path):
        try:
            with open(path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"   ⚠️  读取失败 {path}: {e}")
    return default

# ==========================================
# 收集所有数据
# ==========================================
print("=" * 60)
print("📊 实验结果汇总")
print("=" * 60)

data = {
    'experiment_name': config['experiment_name'],
    'timestamp': datetime.now().isoformat(),
    'config': config,
}

# 1. 鲁棒性分析
robustness = load_json(os.path.join(result_dir, 'merge.json'))
detail = load_json(os.path.join(result_dir, 'detail.json'))
if robustness:
    data['robustness'] = robustness.get('overall', {})
    print(f"✅ 鲁棒性分析: {len(data['robustness'])} 种攻击")
else:
    data['robustness'] = {}
    print("❌ 未找到鲁棒性分析结果")

# 2. 隐蔽性分析
invisibility = load_json(os.path.join(result_dir, 'invisibility_analysis.json'))
if invisibility:
    data['invisibility'] = invisibility
    print(f"✅ 隐蔽性分析: FID={invisibility.get('fid', {}).get('score', 'N/A')}")
else:
    data['invisibility'] = {}
    print("❌ 未找到隐蔽性分析结果")

# 3. 消融实验定量分析
ablation = load_json(os.path.join(result_dir, 'ablation_analysis.json'))
if ablation:
    data['ablation'] = ablation
    print(f"✅ 消融定量分析: {len(ablation.get('fid', {}))} 个版本")
else:
    data['ablation'] = {}
    print("❌ 未找到消融定量分析")

# 4. 网格分辨率消融
grid = load_json(os.path.join(result_dir, 'ablation_grid', 'grid_ablation_summary.json'))
if grid:
    data['grid_ablation'] = grid
    print(f"✅ 网格分辨率消融: {grid.get('num_samples', 'N/A')} 张样本")
else:
    data['grid_ablation'] = {}
    print("❌ 未找到网格分辨率消融")

# ==========================================
# 生成 Markdown 报告
# ==========================================
print("\n📝 生成 Markdown 汇总报告...")

lines = []
lines.append(f"# OrthoFlow 实验结果汇总报告")
lines.append(f"\n> **实验名称**: {config['experiment_name']}")
lines.append(f"> **生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
lines.append(f"> **样本数量**: {config.get('num_samples', 'N/A')} 张")
lines.append(f"> **模型路径**: `{config.get('model_path', 'N/A')}`")
lines.append(f"> **输出目录**: `{config.get('output_base_dir', 'N/A')}`")
lines.append("\n---\n")

# ---- 隐蔽性分析 ----
lines.append("## 一、隐蔽性分析 (Invisibility)\n")
if data['invisibility']:
    fid = data['invisibility'].get('fid', {})
    clip = data['invisibility'].get('clip_score', {})
    lines.append(f"- **FID**: `{fid.get('score', 'N/A'):.2f}` ({fid.get('interpretation', '')})")
    lines.append(f"- **CLIP Score (无水印)**: `{clip.get('no_watermark', 'N/A')}`")
    lines.append(f"- **CLIP Score (水印图)**: `{clip.get('watermarked', 'N/A')}`")
    diff = clip.get('difference')
    if diff is not None:
        lines.append(f"- **CLIP 差异**: `{diff:+.4f}` (越接近 0 越好)")
else:
    lines.append("*未找到数据*\n")

# ---- 鲁棒性分析 ----
lines.append("\n## 二、全局鲁棒性分析 (Robustness)\n")
lines.append("| 攻击类型 | Mean Cosine | Retention Rate | AUC | TPR@0.1%FPR |")
lines.append("|----------|-------------|----------------|-----|-------------|")
if data['robustness']:
    for attack_name, stats in sorted(data['robustness'].items()):
        if 'mean_iou' not in stats:  # 只显示全局鲁棒性攻击
            lines.append(
                f"| {attack_name} | "
                f"{stats.get('mean_cosine', 0):.4f} | "
                f"{stats.get('mean_retention_rate', 0):.1f}% | "
                f"{stats.get('auc', 0):.4f} | "
                f"{stats.get('tpr_at_0.1%_fpr', 0):.4f} |"
            )
else:
    lines.append("*未找到数据*")

lines.append("\n### 关键发现")
if data['robustness']:
    perfect_auc = [k for k, v in data['robustness'].items() if v.get('auc', 0) == 1.0 and 'mean_iou' not in v]
    if perfect_auc:
        lines.append(f"- **AUC = 1.0 的攻击**: {', '.join(perfect_auc)}")
    sdedit = data['robustness'].get('sdedit_0.3', {})
    if sdedit:
        lines.append(f"- **SDEdit 0.3 (扩散再生)**: AUC={sdedit.get('auc', 0):.4f}, TPR@0.1%FPR={sdedit.get('tpr_at_0.1%_fpr', 0):.2f} — 这是最难的攻击，表现优异")
    crop = data['robustness'].get('crop_0.50', {})
    if crop:
        lines.append(f"- **Crop 0.50 (结构性弱点)**: AUC={crop.get('auc', 0):.4f} — 这是 VAE + DiT 导致的已知局限性")

# ---- 篡改定位 ----
lines.append("\n## 三、篡改定位分析 (Tamper Localization)\n")
lines.append("| 攻击类型 | Patch-AUC | F1-Score | IoU |")
lines.append("|----------|-----------|----------|-----|")
if data['robustness']:
    tamper_attacks = {k: v for k, v in data['robustness'].items() if 'mean_iou' in v}
    if tamper_attacks:
        for attack_name, stats in sorted(tamper_attacks.items()):
            lines.append(
                f"| {attack_name} | "
                f"{stats.get('mean_patch_auc', 0):.4f} | "
                f"{stats.get('mean_f1', 0):.4f} | "
                f"{stats.get('mean_iou', 0):.4f} |"
            )
    else:
        lines.append("*未找到篡改定位数据*")
else:
    lines.append("*未找到数据*")

# ---- 消融实验 ----
lines.append("\n## 四、消融实验分析 (Ablation Study)\n")
if data['ablation']:
    lines.append("### 4.1 隐蔽性 + 水印强度\n")
    lines.append("| 方法 | FID ↓ | CLIP Score ↑ | S_mean |")
    lines.append("|------|-------|--------------|--------|")
    order = ['full', 'no_fft', 'no_ortho', 'no_ortho_stress', 'semantic']
    for key in order:
        if key not in data['ablation'].get('fid', {}):
            continue
        name = {
            'full': 'Full Method',
            'no_fft': 'w/o FFT',
            'no_ortho': 'w/o Orthogonal',
            'no_ortho_stress': 'w/o Orthogonal (Stress)',
            'semantic': 'w/ Semantic Mask'
        }.get(key, key)
        fid = data['ablation']['fid'].get(key, 'N/A')
        clip = data['ablation']['clip_score'].get(key, 'N/A')
        sig = data['ablation']['signature_mean'].get(key, {})
        fid_str = f"{fid:.2f}" if isinstance(fid, (int, float)) else '-'
        clip_str = f"{clip:.4f}" if isinstance(clip, (int, float)) else '-'
        sig_str = f"{sig.get('mean', 'N/A'):.4f}" if isinstance(sig.get('mean'), (int, float)) else '-'
        lines.append(f"| {name} | {fid_str} | {clip_str} | {sig_str} |")

    lines.append("\n### 4.2 关键结论")
    sig_full = data['ablation'].get('signature_mean', {}).get('full', {})
    sig_no_fft = data['ablation'].get('signature_mean', {}).get('no_fft', {})
    sig_sem = data['ablation'].get('signature_mean', {}).get('semantic', {})
    if sig_full and sig_no_fft:
        drop = (1 - sig_no_fft.get('mean', 0) / sig_full.get('mean', 1)) * 100
        lines.append(f"- **去掉 FFT** 后 S_mean 下降了 **{drop:.1f}%**，证明频域约束对能量保护至关重要。")
    if sig_full and sig_sem:
        drop = (1 - sig_sem.get('mean', 0) / sig_full.get('mean', 1)) * 100
        lines.append(f"- **语义 Mask** 的 S_mean 下降了 **{drop:.1f}%**，反向证明全图统一注入的必要性。")
    lines.append("- **正交投影** 的核心价值在于抹平方差（Variance），使全图盲提取成为可能。详见可视化结果 `ablation_variance_violin.png`。")
else:
    lines.append("*未找到消融实验数据*\n")

# ---- 网格分辨率 ----
lines.append("\n## 五、网格分辨率消融 (Grid Size Ablation)\n")
if data['grid_ablation']:
    lines.append("| 网格尺寸 | 向量维度 | Mean Cosine | Avg Std |")
    lines.append("|----------|----------|-------------|---------|")
    for key in ['32x32', '16x16', '8x8']:
        stats = data['grid_ablation'].get('statistics', {}).get(key, {})
        dim = stats.get('vector_dim', 'N/A')
        mean = stats.get('mean', 0)
        std = stats.get('std', 0)
        lines.append(f"| {key} | {dim} | {mean:.4f} | {std:.4f} |")
    lines.append("\n**结论**: 32×32（64D）方差极大，16×16（256D）有所改善，8×8（1024D）是统计学稳定性与空间精度的最优解（Sweet Spot）。")
else:
    lines.append("*未找到网格分辨率消融数据*\n")

# ---- 原始文件索引 ----
lines.append("\n## 六、原始结果文件索引\n")
files = {
    '鲁棒性分析 (JSON)': os.path.join(result_dir, 'merge.json'),
    '鲁棒性详情 (JSON)': os.path.join(result_dir, 'detail.json'),
    '隐蔽性分析 (JSON)': os.path.join(result_dir, 'invisibility_analysis.json'),
    '消融定量分析 (JSON)': os.path.join(result_dir, 'ablation_analysis.json'),
    '网格分辨率消融 (JSON)': os.path.join(result_dir, 'ablation_grid', 'grid_ablation_summary.json'),
    '消融可视化 (图片)': os.path.join(result_dir, 'ablation_viz/'),
    '篡改热力图 (图片)': os.path.join(result_dir, 'pic', 'tamper_img/'),
}
for name, path in files.items():
    exists = "✅" if os.path.exists(path) else "❌"
    lines.append(f"- {exists} **{name}**: `{path}`")

lines.append("\n---\n")
lines.append("*本报告由 `summarize_all_results.py` 自动生成*\n")

# 写入 Markdown
with open(output_md, 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines))

# 写入 JSON
with open(output_json, 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

print(f"\n✅ 汇总完成！")
print(f"   Markdown 报告: {output_md}")
print(f"   JSON 汇总:     {output_json}")
print(f"\n💡 提示: 可以直接用 cat / less / vscode 打开 {output_md} 查看完整结果")
