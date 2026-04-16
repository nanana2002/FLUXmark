#!/usr/bin/env python3
"""
鲁棒性分析脚本（极致性能版 - 50GB显存全速运行）
读取预计算的签名文件，进行分析


优化策略：
- 使用GPU加速计算
- 批量处理签名数据
- 并行生成热力图
"""


import os
import json


# 先读取配置设置 GPU（必须在 import torch 之前）
with open('config.json', 'r') as f:
    config = json.load(f)
os.environ['CUDA_VISIBLE_DEVICES'] = str(config.get('gpu_id', 0))


import numpy as np
from datetime import datetime
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, f1_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2
import torch
from PIL import Image


# 路径设置
result_dir = os.path.join(config['output_base_dir'], 'result')
pic_result_dir = os.path.join(result_dir, 'pic', 'tamper_img')
os.makedirs(pic_result_dir, exist_ok=True)


watermark_extra_dir = os.path.join(config['output_base_dir'], 'watermark_extra')
attack_dir = os.path.join(config['output_base_dir'], 'pic', 'attack_watermarked_img')
watermarked_dir = os.path.join(config['output_base_dir'], 'pic', 'watermarked_img')


# GPU加速标志
USE_GPU = torch.cuda.is_available()
if USE_GPU:
    print(f"✅ 使用GPU加速计算")
else:
    print(f"⚠️  使用CPU计算")


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
# 2. 辅助函数（GPU加速版本）
# ==========================================


def compute_iou_gpu(pred_mask, true_mask):
    """使用GPU计算IoU"""
    if USE_GPU:
        pred_tensor = torch.from_numpy(pred_mask.astype(np.float32)).cuda()
        true_tensor = torch.from_numpy(true_mask.astype(np.float32)).cuda()
        intersection = (pred_tensor * true_tensor).sum()
        union = ((pred_tensor + true_tensor) > 0).float().sum()
        return float((intersection / (union + 1e-8)).cpu().numpy())
    else:
        return compute_iou_cpu(pred_mask, true_mask)


def compute_iou_cpu(pred_mask, true_mask):
    """CPU版本IoU"""
    pred_mask = pred_mask.astype(bool)
    true_mask = true_mask.astype(bool)
    intersection = np.logical_and(pred_mask, true_mask).sum()
    union = np.logical_or(pred_mask, true_mask).sum()
    return float(intersection / (union + 1e-8))




def compute_f1_score_gpu(pred_mask, true_mask):
    """使用GPU计算F1分数"""
    if USE_GPU:
        pred_tensor = torch.from_numpy(pred_mask.astype(np.float32)).cuda()
        true_tensor = torch.from_numpy(true_mask.astype(np.float32)).cuda()
        tp = (pred_tensor * true_tensor).sum()
        fp = (pred_tensor * (1 - true_tensor)).sum()
        fn = ((1 - pred_tensor) * true_tensor).sum()


        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        return float(f1.cpu().numpy()), float(precision.cpu().numpy()), float(recall.cpu().numpy())
    else:
        return compute_f1_score_cpu(pred_mask, true_mask)


def compute_f1_score_cpu(pred_mask, true_mask):
    """CPU版本F1"""
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




def compute_smooth_heatmap(S_orig, S_tamp, threshold_ratio=0.8, noise_gate=0.04, img_size=(512, 512)):
    """生成高分辨率平滑篡改热力图（全图自适应阈值）"""
    diff = S_orig - S_tamp
    base_heatmap = np.abs(diff)


    if np.max(base_heatmap) < noise_gate:
        return np.zeros(img_size), np.zeros(img_size, dtype=bool), np.zeros(img_size, dtype=bool), noise_gate


    # 平滑热力图仅用于可视化叠加
    smooth_heatmap = cv2.resize(base_heatmap, img_size, interpolation=cv2.INTER_LINEAR)
    smooth_heatmap = cv2.GaussianBlur(smooth_heatmap, (15, 15), 0)


    # 使用全图统计计算自适应阈值
    base_threshold = np.mean(base_heatmap) + threshold_ratio * np.std(base_heatmap)
    threshold = max(base_threshold, noise_gate)


    # 可视化用的平滑 mask（宽松一点，用于 aesthetic overlay）
    smooth_mask = (smooth_heatmap > threshold).astype(np.uint8)
    kernel = np.ones((5, 5), np.uint8)
    cleaned_mask = cv2.morphologyEx(smooth_mask, cv2.MORPH_OPEN, kernel).astype(bool)


    # 与 IoU 计算严格一致的 raw mask：先在 8x8 上二值化，再用 NEAREST 上采样
    raw_mask_8x8 = (base_heatmap > threshold).astype(np.uint8)
    aligned_mask = cv2.resize(raw_mask_8x8, img_size, interpolation=cv2.INTER_NEAREST).astype(bool)


    return smooth_heatmap, cleaned_mask, aligned_mask, threshold




def visualize_tamper_heatmap(img_orig, img_attacked, S_orig, S_tamp, true_mask,
                              save_path, attack_name, metrics=None):
    """可视化篡改热力图"""
    img_size = img_orig.size if hasattr(img_orig, 'size') else (512, 512)
    if isinstance(img_size, tuple) and len(img_size) == 2:
        img_size = img_size
    else:
        img_size = (512, 512)


    diff = S_orig - S_tamp
    heatmap_8x8 = np.abs(diff)


    smooth_heatmap, cleaned_mask, aligned_mask, threshold = compute_smooth_heatmap(
        S_orig, S_tamp, threshold_ratio=0.8, img_size=img_size
    )


    fig, axes = plt.subplots(2, 4, figsize=(20, 10))


    axes[0, 0].imshow(img_orig)
    axes[0, 0].set_title('Original Watermarked')
    axes[0, 0].axis('off')


    axes[0, 1].imshow(img_attacked)
    axes[0, 1].set_title(f'Attacked: {attack_name}')
    axes[0, 1].axis('off')


    im1 = axes[0, 2].imshow(heatmap_8x8, cmap='hot', interpolation='nearest')
    axes[0, 2].set_title('8x8 Raw Heatmap')
    axes[0, 2].axis('off')
    plt.colorbar(im1, ax=axes[0, 2], fraction=0.046, pad=0.04)


    if true_mask is not None:
        axes[0, 3].imshow(true_mask, cmap='Reds', interpolation='nearest')
        axes[0, 3].set_title('True Tampered Regions')
    else:
        axes[0, 3].text(0.5, 0.5, 'No Ground Truth', ha='center', va='center')
        axes[0, 3].set_title('True Mask (N/A)')
    axes[0, 3].axis('off')


    im2 = axes[1, 0].imshow(smooth_heatmap, cmap='hot', interpolation='bilinear')
    axes[1, 0].set_title(f'High-Res Smooth Heatmap\n({img_size[0]}x{img_size[1]})')
    axes[1, 0].axis('off')
    plt.colorbar(im2, ax=axes[1, 0], fraction=0.046, pad=0.04)


    img_attacked_resized = img_attacked.resize(img_size, Image.LANCZOS)
    axes[1, 1].imshow(img_attacked_resized, alpha=0.6)
    im3 = axes[1, 1].imshow(smooth_heatmap, cmap='hot', alpha=0.5, interpolation='bilinear')
    axes[1, 1].set_title('Heatmap Overlay on Image')
    axes[1, 1].axis('off')


    axes[1, 2].imshow(aligned_mask, cmap='Blues', interpolation='nearest')
    axes[1, 2].set_title('Detected Regions (Aligned w/ IoU)')
    axes[1, 2].axis('off')


    comparison = np.zeros((*aligned_mask.shape, 3))
    if true_mask is not None:
        true_mask_hr = cv2.resize(true_mask.astype(np.uint8), img_size, interpolation=cv2.INTER_NEAREST).astype(bool)


        tp = np.logical_and(aligned_mask, true_mask_hr)
        fp = np.logical_and(aligned_mask, ~true_mask_hr)
        fn = np.logical_and(~aligned_mask, true_mask_hr)
        comparison[tp] = [0, 1, 0]
        comparison[fp] = [1, 1, 0]
        comparison[fn] = [1, 0, 0]
        title = 'Detection Comparison (Aligned w/ IoU)\n(G=TP, Y=FP, R=FN)'
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


    fig, axes = plt.subplots(1, 3, figsize=(18, 6))


    axes[0].imshow(img_attacked)
    axes[0].set_title('Attacked Image')
    axes[0].axis('off')


    axes[1].imshow(smooth_heatmap, cmap='hot', interpolation='bilinear')
    axes[1].set_title(f'Smooth Heatmap ({img_size[0]}x{img_size[1]})')
    axes[1].axis('off')
    plt.colorbar(axes[1].images[0], ax=axes[1], fraction=0.046, pad=0.04)


    axes[2].imshow(img_attacked_resized, alpha=0.5)
    axes[2].imshow(smooth_heatmap, cmap='hot', alpha=0.5, interpolation='bilinear')
    axes[2].set_title('Overlay')
    axes[2].axis('off')


    plt.tight_layout()
    save_path_hr = save_path.replace('.png', '_highres.png')
    plt.savefig(save_path_hr, dpi=200, bbox_inches='tight')
    plt.close()


# ==========================================
# 3. 加载负样本签名（无水印生成图 + 真实自然图像）
# ==========================================
print("\n🔍 加载负样本签名...")
no_watermark_scores = []


no_wm_files = [f for f in os.listdir(watermark_extra_dir) if f.endswith('_S_no_wm.npy')]
for f in tqdm(no_wm_files, desc="加载无水印负样本"):
    S = np.load(os.path.join(watermark_extra_dir, f))
    no_watermark_scores.append(float(np.mean(S)))


real_image_scores = []
real_files = [f for f in os.listdir(watermark_extra_dir) if f.endswith('_S_real.npy')]
for f in tqdm(real_files, desc="加载真实图像负样本"):
    S = np.load(os.path.join(watermark_extra_dir, f))
    real_image_scores.append(float(np.mean(S)))


negative_scores = no_watermark_scores + real_image_scores


print(f"   无水印负样本数量: {len(no_watermark_scores)}")
print(f"   真实图像负样本数量: {len(real_image_scores)}")
print(f"   总负样本数量: {len(negative_scores)}")
if negative_scores:
    print(f"   负样本均值: {np.mean(negative_scores):.4f}")


# ==========================================
# 4. 分析攻击结果（高性能模式）
# ==========================================
print("\n🔬 开始分析攻击结果（高性能模式）...")


detail_results = []
merge_results = {}


for attack_name in tqdm(attack_configs, desc="分析攻击"):
    attack_results = {
        'attack_name': attack_name,
        'images': [],
        'S_orig_means': [],
        'S_tamp_means': [],
        'retention_rates': []
    }


    if any(k in attack_name for k in ['black_block', 'inpaint', 'fluxfill', 'copy_move', 'splicing']):
        attack_results['ious'] = []
        attack_results['f1s'] = []
        attack_results['patch_aucs'] = []


    for img_info in images_info:
        img_id = img_info['id']
        prompt = img_info['prompt']


        S_orig_path = os.path.join(watermark_extra_dir, f"{img_id}_S_orig.npy")
        if not os.path.exists(S_orig_path):
            continue


        S_orig = np.load(S_orig_path)


        # 支持子文件夹和根目录两种存放方式
        S_tamp_path = os.path.join(watermark_extra_dir, attack_name, f"{img_id}_{attack_name}_S_tamp.npy")
        if not os.path.exists(S_tamp_path):
            S_tamp_path = os.path.join(watermark_extra_dir, f"{img_id}_{attack_name}_S_tamp.npy")
        if not os.path.exists(S_tamp_path):
            continue


        S_tamp = np.load(S_tamp_path)


        S_orig_mean = float(np.mean(S_orig))
        S_tamp_mean = float(np.mean(S_tamp))
        retention_rate = compute_retention_rate(S_tamp_mean, S_orig_mean)


        img_result = {
            'img_id': img_id,
            'S_orig_mean': S_orig_mean,
            'S_tamp_mean': S_tamp_mean,
            'retention_rate': retention_rate
        }


        if any(k in attack_name for k in ['black_block', 'inpaint', 'fluxfill', 'copy_move', 'splicing']):
            true_mask_path = os.path.join(attack_dir, attack_name, f"{img_id}_mask.npy")
            if os.path.exists(true_mask_path):
                true_mask = np.load(true_mask_path)


                diff = S_orig - S_tamp
                heatmap = np.abs(diff)


                y_true_flat = true_mask.flatten()
                y_score_flat = heatmap.flatten()
                try:
                    patch_auc = roc_auc_score(y_true_flat, y_score_flat)
                except:
                    patch_auc = 0.5


                threshold = np.mean(heatmap) + 0.5 * np.std(heatmap)
                pred_mask = (heatmap > threshold).astype(int)


                # 使用GPU计算
                if USE_GPU:
                    iou = compute_iou_gpu(pred_mask, true_mask)
                    f1, precision, recall = compute_f1_score_gpu(pred_mask, true_mask)
                else:
                    iou = compute_iou_cpu(pred_mask, true_mask)
                    f1, precision, recall = compute_f1_score_cpu(pred_mask, true_mask)


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

                # 修改输出图
                if len(attack_results['images']) == 27:
                    orig_img_path = os.path.join(watermarked_dir, img_info['image_file'])
                    attacked_img_path = os.path.join(attack_dir, attack_name, f"{img_id}.png")
                    if not os.path.exists(attacked_img_path):
                        attacked_img_path = os.path.join(attack_dir, attack_name, f"{img_id}.jpg")


                    if os.path.exists(orig_img_path) and os.path.exists(attacked_img_path):
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


    if attack_results['S_tamp_means']:
        merge_results[attack_name] = {
            'mean_cosine': float(np.mean(attack_results['S_tamp_means'])),
            'std_cosine': float(np.std(attack_results['S_tamp_means'])),
            'mean_retention_rate': float(np.mean(attack_results['retention_rates']))
        }


        if negative_scores:
            y_true = [1] * len(attack_results['S_tamp_means']) + [0] * len(negative_scores)
            y_scores = attack_results['S_tamp_means'] + negative_scores


            try:
                auc = roc_auc_score(y_true, y_scores)
                merge_results[attack_name]['auc'] = float(auc)
            except:
                merge_results[attack_name]['auc'] = 0.5


            try:
                tpr_at_fpr, threshold = compute_tpr_at_fpr(
                    attack_results['S_tamp_means'],
                    negative_scores,
                    target_fpr=0.001
                )
                merge_results[attack_name]['tpr_at_0.1%_fpr'] = float(tpr_at_fpr)
            except:
                merge_results[attack_name]['tpr_at_0.1%_fpr'] = 0.0


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
        'negative_baseline': {
            'no_wm_mean': float(np.mean(no_watermark_scores)) if no_watermark_scores else 0,
            'no_wm_std': float(np.std(no_watermark_scores)) if no_watermark_scores else 0,
            'real_mean': float(np.mean(real_image_scores)) if real_image_scores else 0,
            'real_std': float(np.std(real_image_scores)) if real_image_scores else 0,
            'combined_mean': float(np.mean(negative_scores)) if negative_scores else 0,
            'combined_std': float(np.std(negative_scores)) if negative_scores else 0
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


if USE_GPU:
    print(f"\n🚀 高性能模式：使用GPU加速计算！")
print(f"   最终显存使用: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
print("\n✅ 鲁棒性分析完成！")






