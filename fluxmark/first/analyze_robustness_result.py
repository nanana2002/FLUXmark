#!/usr/bin/env python3
"""
鲁棒性分析脚本（纯分析版，不加载模型）
读取预计算的签名文件，进行分析

前置步骤:
    1. 运行 watermarked_img_generate.py 生成水印图像
    2. 运行 attack_img_generate.py 生成攻击图像
    3. 运行 extract_watermarks.py 提取所有签名
    4. 运行本脚本进行分析
"""

import numpy as np
import os
import json
from datetime import datetime
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, f1_score
import matplotlib.pyplot as plt
import cv2  # 用于高分辨率热力图生成

# 加载配置
with open('config.json', 'r') as f:
    config = json.load(f)

# 路径设置
result_dir = os.path.join(config['output_base_dir'], 'result')
pic_result_dir = os.path.join(result_dir, 'pic', 'tamper_img')
os.makedirs(pic_result_dir, exist_ok=True)

watermark_extra_dir = os.path.join(config['output_base_dir'], 'watermark_extra')
attack_dir = os.path.join(config['output_base_dir'], 'pic', 'attack_watermarked_img')
watermarked_dir = os.path.join(config['output_base_dir'], 'pic', 'watermarked_img')

# ==========================================
# 1. 加载提取摘要
# ==========================================
print("📋 加载签名提取摘要...")
extraction_summary_path = os.path.join(watermark_extra_dir, 'extraction_summary.json')

if not os.path.exists(extraction_summary_path):
    print("❌ 未找到提取摘要文件！")
    print("   请先运行: python3 extract_watermarks.py")
    exit(1)

with open(extraction_summary_path, 'r') as f:
    extraction_summary = json.load(f)

attack_configs = extraction_summary['attack_types']
print(f"   攻击类型: {len(attack_configs)} 种")
print(f"   图像数量: {extraction_summary['num_images']} 张")

# 加载水印摘要获取prompts
with open(os.path.join(watermarked_dir, f'{config["experiment_name"]}_watermark_summary.json'), 'r') as f:
    watermark_summary = json.load(f)

images_info = watermark_summary['images']

# ==========================================
# 2. 辅助函数
# ==========================================

def compute_iou(pred_mask, true_mask):
    """计算IoU（交并比）"""
    pred_mask = pred_mask.astype(bool)
    true_mask = true_mask.astype(bool)
    intersection = np.logical_and(pred_mask, true_mask).sum()
    union = np.logical_or(pred_mask, true_mask).sum()
    return float(intersection / (union + 1e-8))


def compute_f1_score(pred_mask, true_mask):
    """计算F1分数"""
    pred_mask = pred_mask.astype(bool)
    true_mask = true_mask.astype(bool)
    tp = np.logical_and(pred_mask, true_mask).sum()
    fp = np.logical_and(pred_mask, ~true_mask).sum()
    fn = np.logical_and(~pred_mask, true_mask).sum()
    
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    return float(f1), float(precision), float(recall)


def compute_tpr_at_fpr(positive_scores, negative_scores, target_fpr=0.001):
    """计算在给定FPR下的TPR"""
    threshold = np.percentile(negative_scores, (1 - target_fpr) * 100)
    tpr = np.mean(np.array(positive_scores) > threshold)
    return float(tpr), float(threshold)


def compute_retention_rate(S_tamp_mean, S_orig_mean):
    """计算信号保留率"""
    return S_tamp_mean / (S_orig_mean + 1e-8) * 100


def compute_smooth_heatmap(S_orig, S_tamp, threshold_ratio=1.5, noise_gate=0.04, img_size=(512, 512)):
    """
    生成高分辨率平滑篡改热力图
    使用双三次插值上采样 + 高斯模糊
    """
    import cv2
    
    # 1. 计算基础差分 (8x8)
    diff = S_orig - S_tamp
    base_heatmap = np.abs(diff)
    
    # 2. 底噪门控
    if np.max(base_heatmap) < noise_gate:
        return np.zeros(img_size), np.zeros(img_size, dtype=bool), noise_gate
    
    # 🌟 视觉升级 1：双三次插值上采样
    smooth_heatmap = cv2.resize(base_heatmap, img_size, interpolation=cv2.INTER_CUBIC)
    
    # 🌟 视觉升级 2：高斯模糊融合
    smooth_heatmap = cv2.GaussianBlur(smooth_heatmap, (21, 21), 0)
    
    # 3. 计算自适应阈值（使用8x8核心区域）
    mask_8x8 = np.zeros((8, 8))
    mask_8x8[1:7, 1:7] = 1.0
    core_mask = mask_8x8 == 1.0
    core_diff = base_heatmap[core_mask]
    
    if len(core_diff) > 0:
        base_threshold = np.mean(core_diff) + threshold_ratio * np.std(core_diff)
    else:
        base_threshold = np.mean(base_heatmap) + threshold_ratio * np.std(base_heatmap)
    
    threshold = max(base_threshold, noise_gate)
    
    # 4. 生成全尺寸二值掩码
    raw_mask = smooth_heatmap > threshold
    
    # 5. 形态学清理
    kernel = np.ones((5, 5), np.uint8)
    cleaned_mask = cv2.morphologyEx(raw_mask.astype(np.uint8), cv2.MORPH_OPEN, kernel).astype(bool)
    
    return smooth_heatmap, cleaned_mask, threshold


def visualize_tamper_heatmap(img_orig, img_attacked, S_orig, S_tamp, true_mask, 
                              save_path, attack_name, metrics=None):
    """可视化篡改热力图（包含高分辨率版本）"""
    
    # 获取图像尺寸
    img_size = img_orig.size if hasattr(img_orig, 'size') else (512, 512)
    if isinstance(img_size, tuple) and len(img_size) == 2:
        img_size = img_size
    else:
        img_size = (512, 512)
    
    # 计算基础差分热力图 (8x8)
    diff = S_orig - S_tamp
    heatmap_8x8 = np.abs(diff)
    
    # 计算高分辨率热力图
    smooth_heatmap, cleaned_mask, threshold = compute_smooth_heatmap(
        S_orig, S_tamp, threshold_ratio=1.5, img_size=img_size
    )
    
    # 创建大图 (2行4列)
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    
    # Row 1: 基础分析
    # 原图
    axes[0, 0].imshow(img_orig)
    axes[0, 0].set_title('Original Watermarked')
    axes[0, 0].axis('off')
    
    # 攻击后图像
    axes[0, 1].imshow(img_attacked)
    axes[0, 1].set_title(f'Attacked: {attack_name}')
    axes[0, 1].axis('off')
    
    # 8x8 差分热力图（原始）
    im1 = axes[0, 2].imshow(heatmap_8x8, cmap='hot', interpolation='nearest')
    axes[0, 2].set_title('8x8 Raw Heatmap')
    axes[0, 2].axis('off')
    plt.colorbar(im1, ax=axes[0, 2], fraction=0.046, pad=0.04)
    
    # 真实篡改区域
    if true_mask is not None:
        axes[0, 3].imshow(true_mask, cmap='Reds', interpolation='nearest')
        axes[0, 3].set_title('True Tampered Regions')
    else:
        axes[0, 3].text(0.5, 0.5, 'No Ground Truth', ha='center', va='center')
        axes[0, 3].set_title('True Mask (N/A)')
    axes[0, 3].axis('off')
    
    # Row 2: 高分辨率分析
    # 高分辨率平滑热力图
    im2 = axes[1, 0].imshow(smooth_heatmap, cmap='hot', interpolation='bilinear')
    axes[1, 0].set_title(f'High-Res Smooth Heatmap\n({img_size[0]}x{img_size[1]})')
    axes[1, 0].axis('off')
    plt.colorbar(im2, ax=axes[1, 0], fraction=0.046, pad=0.04)
    
    # 热力图叠加在原图上
    axes[1, 1].imshow(img_attacked, alpha=0.6)
    im3 = axes[1, 1].imshow(smooth_heatmap, cmap='hot', alpha=0.5, interpolation='bilinear')
    axes[1, 1].set_title('Heatmap Overlay on Image')
    axes[1, 1].axis('off')
    
    # 检测到的篡改区域（高分辨率）
    axes[1, 2].imshow(cleaned_mask, cmap='Blues', interpolation='nearest')
    axes[1, 2].set_title('Detected Regions (High-Res)')
    axes[1, 2].axis('off')
    
    # 对比图
    comparison = np.zeros((*cleaned_mask.shape, 3))
    if true_mask is not None:
        # 上采样true_mask到高分辨率
        import cv2
        true_mask_hr = cv2.resize(true_mask.astype(np.uint8), img_size, interpolation=cv2.INTER_NEAREST).astype(bool)
        
        tp = np.logical_and(cleaned_mask, true_mask_hr)
        fp = np.logical_and(cleaned_mask, ~true_mask_hr)
        fn = np.logical_and(~cleaned_mask, true_mask_hr)
        comparison[tp] = [0, 1, 0]  # 绿色：正确
        comparison[fp] = [1, 1, 0]  # 黄色：误检
        comparison[fn] = [1, 0, 0]  # 红色：漏检
        title = 'Detection Comparison (High-Res)\n(G=TP, Y=FP, R=FN)'
    else:
        title = 'Predicted Mask (High-Res)'
    axes[1, 3].imshow(comparison, interpolation='nearest')
    if metrics:
        title += f"\nIoU={metrics.get('iou', 0):.3f}, F1={metrics.get('f1', 0):.3f}"
    axes[1, 3].set_title(title)
    axes[1, 3].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # 额外保存高分辨率热力图单独版本
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    axes[0].imshow(img_attacked)
    axes[0].set_title('Attacked Image')
    axes[0].axis('off')
    
    axes[1].imshow(smooth_heatmap, cmap='hot', interpolation='bilinear')
    axes[1].set_title(f'Smooth Heatmap ({img_size[0]}x{img_size[1]})')
    axes[1].axis('off')
    plt.colorbar(axes[1].images[0], ax=axes[1], fraction=0.046, pad=0.04)
    
    axes[2].imshow(img_attacked, alpha=0.5)
    axes[2].imshow(smooth_heatmap, cmap='hot', alpha=0.5, interpolation='bilinear')
    axes[2].set_title('Overlay')
    axes[2].axis('off')
    
    plt.tight_layout()
    save_path_hr = save_path.replace('.png', '_highres.png')
    plt.savefig(save_path_hr, dpi=200, bbox_inches='tight')
    plt.close()

# ==========================================
# 3. 加载无水印图像签名（负样本）
# ==========================================
print("\n🔍 加载无水印图像签名（负样本）...")
no_watermark_scores = []

no_wm_files = [f for f in os.listdir(watermark_extra_dir) if f.endswith('_S_no_wm.npy')]
for f in tqdm(no_wm_files, desc="加载负样本"):
    S = np.load(os.path.join(watermark_extra_dir, f))
    mask_8x8 = np.zeros((8, 8))
    mask_8x8[1:7, 1:7] = 1.0
    core_scores = S[mask_8x8 == 1.0]
    no_watermark_scores.append(float(np.mean(core_scores)))

print(f"   负样本数量: {len(no_watermark_scores)}")
if no_watermark_scores:
    print(f"   负样本均值: {np.mean(no_watermark_scores):.4f}")

# ==========================================
# 4. 分析攻击结果
# ==========================================
print("\n🔬 开始分析攻击结果...")

detail_results = []
merge_results = {}

mask_8x8 = np.zeros((8, 8))
mask_8x8[1:7, 1:7] = 1.0

for attack_name in tqdm(attack_configs, desc="分析攻击"):
    attack_results = {
        'attack_name': attack_name,
        'images': [],
        'S_orig_means': [],
        'S_tamp_means': [],
        'retention_rates': []
    }
    
    # 篡改定位统计
    if 'black_block' in attack_name or 'inpaint' in attack_name:
        attack_results['ious'] = []
        attack_results['f1s'] = []
        attack_results['patch_aucs'] = []
    
    for img_info in images_info:
        img_id = img_info['id']
        prompt = img_info['prompt']
        
        # 加载原始签名 S_orig
        S_orig_path = os.path.join(watermark_extra_dir, f"{img_id}_S_orig.npy")
        if not os.path.exists(S_orig_path):
            continue
        
        S_orig = np.load(S_orig_path)
        
        # 加载攻击后签名 S_tamp
        S_tamp_path = os.path.join(watermark_extra_dir, f"{img_id}_{attack_name}_S_tamp.npy")
        if not os.path.exists(S_tamp_path):
            continue
        
        S_tamp = np.load(S_tamp_path)
        
        # 计算统计
        S_orig_mean = float(np.mean(S_orig[mask_8x8 == 1.0]))
        S_tamp_mean = float(np.mean(S_tamp[mask_8x8 == 1.0]))
        retention_rate = compute_retention_rate(S_tamp_mean, S_orig_mean)
        
        img_result = {
            'img_id': img_id,
            'S_orig_mean': S_orig_mean,
            'S_tamp_mean': S_tamp_mean,
            'retention_rate': retention_rate
        }
        
        # 篡改定位分析
        if 'black_block' in attack_name or 'inpaint' in attack_name:
            true_mask_path = os.path.join(attack_dir, attack_name, f"{img_id}_mask.npy")
            if os.path.exists(true_mask_path):
                true_mask = np.load(true_mask_path)
                
                # 计算差分热力图
                diff = S_orig - S_tamp
                heatmap = np.abs(diff)
                
                # Patch-level AUC
                y_true_flat = true_mask.flatten()
                y_score_flat = heatmap.flatten()
                try:
                    patch_auc = roc_auc_score(y_true_flat, y_score_flat)
                except:
                    patch_auc = 0.5
                
                # 自适应阈值
                threshold = np.mean(heatmap) + 0.5 * np.std(heatmap)
                pred_mask = (heatmap > threshold).astype(int)
                
                iou = compute_iou(pred_mask, true_mask)
                f1, precision, recall = compute_f1_score(pred_mask, true_mask)
                
                img_result.update({
                    'iou': iou,
                    'f1': f1,
                    'precision': precision,
                    'recall': recall,
                    'patch_auc': patch_auc
                })
                
                attack_results['ious'].append(iou)
                attack_results['f1s'].append(f1)
                attack_results['patch_aucs'].append(patch_auc)
                
                # 生成可视化（第一张图）
                if len(attack_results['images']) == 0:
                    orig_img_path = os.path.join(watermarked_dir, img_info['image_file'])
                    attacked_img_path = os.path.join(attack_dir, attack_name, f"{img_id}.png")
                    if not os.path.exists(attacked_img_path):
                        attacked_img_path = os.path.join(attack_dir, attack_name, f"{img_id}.jpg")
                    
                    if os.path.exists(orig_img_path) and os.path.exists(attacked_img_path):
                        from PIL import Image
                        orig_img = Image.open(orig_img_path)
                        attacked_img = Image.open(attacked_img_path)
                        viz_path = os.path.join(pic_result_dir, f"{attack_name}_{img_id}_heatmap.png")
                        metrics = {'iou': iou, 'f1': f1}
                        visualize_tamper_heatmap(orig_img, attacked_img, S_orig, S_tamp, 
                                                  true_mask, viz_path, attack_name, metrics)
        
        attack_results['images'].append(img_result)
        attack_results['S_orig_means'].append(S_orig_mean)
        attack_results['S_tamp_means'].append(S_tamp_mean)
        attack_results['retention_rates'].append(retention_rate)
    
    # 计算攻击的整体统计
    if attack_results['S_tamp_means']:
        merge_results[attack_name] = {
            'mean_cosine': float(np.mean(attack_results['S_tamp_means'])),
            'std_cosine': float(np.std(attack_results['S_tamp_means'])),
            'mean_retention_rate': float(np.mean(attack_results['retention_rates']))
        }
        
        # 计算AUC和TPR@FPR
        if no_watermark_scores:
            y_true = [1] * len(attack_results['S_tamp_means']) + [0] * len(no_watermark_scores)
            y_scores = attack_results['S_tamp_means'] + no_watermark_scores
            
            try:
                auc = roc_auc_score(y_true, y_scores)
                merge_results[attack_name]['auc'] = float(auc)
            except:
                merge_results[attack_name]['auc'] = 0.5
            
            try:
                tpr_at_fpr, threshold = compute_tpr_at_fpr(
                    attack_results['S_tamp_means'], 
                    no_watermark_scores, 
                    target_fpr=0.001
                )
                merge_results[attack_name]['tpr_at_0.1%_fpr'] = float(tpr_at_fpr)
            except:
                merge_results[attack_name]['tpr_at_0.1%_fpr'] = 0.0
        
        # 篡改定位统计
        if 'ious' in attack_results and attack_results['ious']:
            merge_results[attack_name]['mean_iou'] = float(np.mean(attack_results['ious']))
            merge_results[attack_name]['mean_f1'] = float(np.mean(attack_results['f1s']))
            merge_results[attack_name]['mean_patch_auc'] = float(np.mean(attack_results['patch_aucs']))
    
    detail_results.append(attack_results)

# ==========================================
# 5. 保存结果
# ==========================================
print("\n💾 保存分析结果...")

detail_path = os.path.join(result_dir, 'detail.json')
with open(detail_path, 'w') as f:
    json.dump({
        'experiment_name': config['experiment_name'],
        'timestamp': datetime.now().isoformat(),
        'no_watermark_baseline': {
            'mean': float(np.mean(no_watermark_scores)) if no_watermark_scores else 0,
            'std': float(np.std(no_watermark_scores)) if no_watermark_scores else 0
        },
        'attacks': detail_results
    }, f, indent=2)

merge_path = os.path.join(result_dir, 'merge.json')
with open(merge_path, 'w') as f:
    json.dump({
        'experiment_name': config['experiment_name'],
        'timestamp': datetime.now().isoformat(),
        'overall': merge_results
    }, f, indent=2)

print(f"   详细结果: {detail_path}")
print(f"   合并结果: {merge_path}")
print(f"   热力图: {pic_result_dir}")

# ==========================================
# 6. 打印汇总表格
# ==========================================
print("\n" + "="*100)
print("📊 Table 1: 全局鲁棒性评价 (Robustness / Copyright Verification)")
print("="*100)
print(f"{'Attack Type':<25} {'Mean Cosine':>15} {'Retention Rate':>18} {'AUC':>10} {'TPR@0.1%FPR':>15}")
print("-"*100)

for attack_name in attack_configs:
    if attack_name not in merge_results:
        continue
    stats = merge_results[attack_name]
    print(f"{attack_name:<25} {stats.get('mean_cosine', 0):>15.4f} "
          f"{stats.get('mean_retention_rate', 0):>17.2f}% "
          f"{stats.get('auc', 0):>10.4f} "
          f"{stats.get('tpr_at_0.1%_fpr', 0):>15.4f}")

print("="*100)

print("\n" + "="*100)
print("📊 Table 2: 篡改定位评价 (Tamper Localization / Fragility)")
print("="*100)
print(f"{'Attack Type':<25} {'Patch-AUC':>15} {'F1-Score':>15} {'IoU':>15}")
print("-"*100)

for attack_name in attack_configs:
    if attack_name not in merge_results:
        continue
    stats = merge_results[attack_name]
    if 'mean_patch_auc' in stats:
        print(f"{attack_name:<25} {stats.get('mean_patch_auc', 0):>15.4f} "
              f"{stats.get('mean_f1', 0):>15.4f} "
              f"{stats.get('mean_iou', 0):>15.4f}")

print("="*100)

print("\n✅ 鲁棒性分析完成！")
