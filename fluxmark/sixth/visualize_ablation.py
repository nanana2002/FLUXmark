#!/usr/bin/env python3
"""
消融实验可视化脚本
生成两类图：
1. 逐样本图像对比（每个 prompt 一排，直观对比画质差异）
2. 签名方差分布图（Boxplot，展示正交投影对 variance 的压制）
"""


import os
import json


with open('config.json', 'r') as f:
    config = json.load(f)


import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm


result_dir = os.path.join(config['output_base_dir'], 'result', 'ablation_viz')
os.makedirs(result_dir, exist_ok=True)


base_pic_dir = os.path.join(config['output_base_dir'], 'pic')
no_wm_dir = os.path.join(base_pic_dir, 'no_watermarked_img')
full_wm_dir = os.path.join(base_pic_dir, 'watermarked_img')


ABLATIONS = {
    'no_wm': {
        'name': 'No Watermark',
        'dir': no_wm_dir,
        'prefix': 'no_wm',
        'color': '#808080'
    },
    'full': {
        'name': 'Full Method',
        'dir': full_wm_dir,
        'prefix': 'wm',
        'color': '#2ecc71'
    },
    'no_fft': {
        'name': 'w/o FFT',
        'dir': os.path.join(base_pic_dir, 'ablation_fft_watermarked_img'),
        'prefix': 'ablation_fft',
        'color': '#3498db'
    },
    'no_ortho': {
        'name': 'w/o Orthogonal',
        'dir': os.path.join(base_pic_dir, 'ablation_obj_watermarked_img'),
        'prefix': 'ablation_obj',
        'color': '#e74c3c'
    },
    'no_ortho_stress': {
        'name': 'w/o Orthogonal (α=2.5)',
        'dir': os.path.join(base_pic_dir, 'ablation_obj_stress_watermarked_img'),
        'prefix': 'ablation_obj_stress',
        'fallback_prefix': 'ablation_obj',  # 兼容旧版本生成的文件名
        'color': '#c0392b'
    },
    'semantic': {
        'name': 'w/ Semantic Mask',
        'dir': os.path.join(base_pic_dir, 'ablation_sem_watermarked_img'),
        'prefix': 'ablation_sem',
        'color': '#9b59b6'
    }
}


# 加载无水印摘要获取 prompts
no_wm_summary_path = os.path.join(no_wm_dir, f'{config["experiment_name"]}_no_watermark_summary.json')
with open(no_wm_summary_path, 'r') as f:
    no_wm_summary = json.load(f)
num_images = len(no_wm_summary['images'])


# 检查哪些消融目录实际存在
available_ablations = {}
for key, info in ABLATIONS.items():
    exists = os.path.exists(info['dir'])
    if not exists:
        print(f"   ⚠️  跳过 {info['name']}: 目录不存在 {info['dir']}")
        continue


    files = os.listdir(info['dir'])
    png_files = [f for f in files if f.endswith('.png')]
    prefix = info['prefix']
    fallback = info.get('fallback_prefix')


    prefix_match = any(f.startswith(prefix) for f in png_files)
    fallback_match = fallback and any(f.startswith(fallback) for f in png_files)


    if prefix_match or fallback_match:
        available_ablations[key] = info
        actual_prefix = prefix if prefix_match else fallback
        matched_count = len([f for f in png_files if f.startswith(actual_prefix)])
        print(f"   ✅ {info['name']}: 找到 {matched_count} 张图 (prefix={actual_prefix})")
    else:
        print(f"   ⚠️  跳过 {info['name']}: 目录中 {len(png_files)} 个 png，但没有以 {prefix} 或 {fallback} 开头的")
        print(f"      目录内容样例: {png_files[:5]}")


if not available_ablations:
    print("❌ 未找到任何消融实验图像，请先运行消融实验生成脚本")
    exit(1)


print(f"✅ 将使用以下版本做对比: {list(available_ablations.keys())}")


# ==========================================
# 1. 逐样本图像对比
# ==========================================
print("\n🎨 生成逐样本对比图...")


num_viz = min(num_images, 20)  # 最多可视化前 20 张
for idx in tqdm(range(num_viz), desc="对比图"):
    methods = list(available_ablations.keys())
    n_methods = len(methods)


    fig, axes = plt.subplots(1, n_methods, figsize=(3.5 * n_methods, 4))
    if n_methods == 1:
        axes = [axes]


    for ax, key in zip(axes, methods):
        info = available_ablations[key]
        img_path = os.path.join(info['dir'], f"{info['prefix']}_{idx:03d}.png")


        if os.path.exists(img_path):
            img = Image.open(img_path).convert('RGB')
            ax.imshow(img)
        else:
            ax.text(0.5, 0.5, 'N/A', ha='center', va='center', transform=ax.transAxes)


        ax.set_title(info['name'], fontsize=12, color=info.get('color', 'black'), fontweight='bold')
        ax.axis('off')


    # 添加总标题（prompt）
    prompt = no_wm_summary['images'][idx]['prompt']
    fig.suptitle(f"Sample {idx:03d}: {prompt[:80]}{'...' if len(prompt) > 80 else ''}",
                 fontsize=14, fontweight='bold', y=1.02)


    plt.tight_layout()
    save_path = os.path.join(result_dir, f"ablation_compare_{idx:03d}.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


print(f"   已生成 {num_viz} 张对比图 -> {result_dir}")


# ==========================================
# 2. 签名方差分布图（Boxplot / Violin）
# ==========================================
print("\n📊 生成签名方差分布图...")


# 收集各版本的签名标准差
variance_data = {}
labels = []
colors = []


for key, info in available_ablations.items():
    if key == 'no_wm':
        continue  # 无水印图没有签名


    summary_path = os.path.join(info['dir'], f'{config["experiment_name"]}_{info["prefix"]}_summary.json')
    if not os.path.exists(summary_path):
        # 尝试旧命名
        alt_name = info['prefix'].replace('ablation_', '')
        summary_path = os.path.join(info['dir'], f'{config["experiment_name"]}_ablation_{alt_name}_summary.json')
    if not os.path.exists(summary_path) and key == 'full':
        # Full Method 的特殊命名：watermark_summary 而非 wm_summary
        summary_path = os.path.join(info['dir'], f'{config["experiment_name"]}_watermark_summary.json')


    if not os.path.exists(summary_path):
        print(f"   ⚠️  {info['name']}: 未找到摘要文件 {summary_path}")
        continue


    with open(summary_path, 'r') as f:
        summary = json.load(f)


    stds = [img['stats']['std'] for img in summary.get('images', []) if 'stats' in img and 'std' in img['stats']]
    if stds:
        variance_data[key] = stds
        labels.append(info['name'])
        colors.append(info.get('color', '#3498db'))


if variance_data:
    fig, ax = plt.subplots(figsize=(10, 6))


    positions = range(1, len(variance_data) + 1)
    data_list = [variance_data[k] for k in variance_data.keys()]


    # Boxplot
    bp = ax.boxplot(data_list, positions=positions, widths=0.5, patch_artist=True,
                    showmeans=True, meanline=True)


    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)


    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=15, ha='right')
    ax.set_ylabel('Signature Std (Lower = More Uniform)', fontsize=12)
    ax.set_title('Ablation Study: Signature Variance Distribution\n(Token-level Orthogonal Projection flattens variance)', fontsize=13, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)


    plt.tight_layout()
    save_path = os.path.join(result_dir, 'ablation_variance_boxplot.png')
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"   已生成方差分布图 -> {save_path}")


    # 同时生成 Violin Plot（更美观）
    fig, ax = plt.subplots(figsize=(10, 6))
    parts = ax.violinplot(data_list, positions=positions, widths=0.7, showmeans=True, showmedians=True)


    for pc, color in zip(parts['bodies'], colors):
        pc.set_facecolor(color)
        pc.set_alpha(0.7)


    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=15, ha='right')
    ax.set_ylabel('Signature Std (Lower = More Uniform)', fontsize=12)
    ax.set_title('Ablation Study: Signature Variance (Violin Plot)\nOrthogonal Drift achieves mirror-like uniformity', fontsize=13, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)


    plt.tight_layout()
    save_path = os.path.join(result_dir, 'ablation_variance_violin.png')
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"   已生成小提琴图 -> {save_path}")


    # 打印统计摘要
    print("\n📈 方差统计摘要:")
    for key, stds in variance_data.items():
        print(f"   {ABLATIONS[key]['name']:<25} Std of Std = {np.mean(stds):.4f} ± {np.std(stds):.4f}  (median={np.median(stds):.4f})")


else:
    print("   ⚠️  未找到消融实验摘要文件，跳过方差分布图")


# ==========================================
# 3. 压力测试特写对比（如果有的话）
# ==========================================
if 'full' in available_ablations and 'no_ortho_stress' in available_ablations:
    print("\n🔥 生成压力测试特写对比...")
    num_stress = min(num_images, 10)


    for idx in tqdm(range(num_stress), desc="压力测试对比"):
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))


        for ax, key in zip(axes, ['no_wm', 'full', 'no_ortho_stress']):
            info = available_ablations[key]
            img_path = os.path.join(info['dir'], f"{info['prefix']}_{idx:03d}.png")


            if os.path.exists(img_path):
                img = Image.open(img_path).convert('RGB')
                ax.imshow(img)
            else:
                ax.text(0.5, 0.5, 'N/A', ha='center', va='center', transform=ax.transAxes)


            title_color = 'black' if key == 'no_wm' else ('green' if key == 'full' else 'red')
            ax.set_title(info['name'], fontsize=12, color=title_color, fontweight='bold')
            ax.axis('off')


        fig.suptitle(f"Stress Test (α=2.5) Sample {idx:03d}", fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        save_path = os.path.join(result_dir, f"stress_compare_{idx:03d}.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()


    print(f"   已生成 {num_stress} 张压力测试特写 -> {result_dir}")


print("\n✅ 消融实验可视化完成！")
print(f"   所有图片保存位置: {result_dir}")






