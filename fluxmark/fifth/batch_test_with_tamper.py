#!/usr/bin/env python3
"""
OrthoFlow 规模化鲁棒性测试 + 篡改定位评估
包含：传统攻击 + 黑色方块篡改 + 篡改定位可视化
"""

import torch
from diffusers import FluxPipeline
import torchvision.transforms as T
from torchvision.transforms import functional as F_vision
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import io
import os
import gc
import json
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as patches

os.environ['HF_HOME'] = '/data/daiyina/hf_cache'
output_dir = '/data/daiyina/project_flux/pic'

# ==========================================
# 1. 加载实验配置
# ==========================================
print("📋 加载实验配置...")
with open('prompts.json', 'r') as f:
    config = json.load(f)

experiment_name = config['experiment_name']
num_runs = 3  # 每种攻击运行次数（减少以节省时间）

print(f"实验名称: {experiment_name}")
print(f"每种攻击运行: {num_runs} 次")

# ==========================================
# 2. 先加载生成摘要获取prompts
# ==========================================
summary_path = os.path.join(output_dir, f'{experiment_name}_summary.json')
with open(summary_path, 'r') as f:
    gen_summary = json.load(f)

images_info = gen_summary['images']
print(f"\n📊 加载了 {len(images_info)} 张图像的生成信息")

# ==========================================
# 3. 加载模型并预编码所有prompts
# ==========================================
print("\n🚀 加载 FLUX 模型...")
pipe = FluxPipeline.from_pretrained(
    '/data/daiyina/project_flux/model/flux-schnell',
    torch_dtype=torch.bfloat16
)

# 预编码所有prompts
print(f"🔤 预编码 {len(images_info)} 个prompts...")
encoded_prompts_cache = {}
for img_info in images_info:
    prompt_text = img_info['prompt']
    if prompt_text not in encoded_prompts_cache:
        with torch.no_grad():
            prompt_embeds, pooled_prompt_embeds, text_ids = pipe.encode_prompt(
                prompt=prompt_text, prompt_2=None, max_sequence_length=256
            )
        encoded_prompts_cache[prompt_text] = {
            'prompt_embeds': prompt_embeds.cpu(),
            'pooled_prompt_embeds': pooled_prompt_embeds.cpu(),
            'text_ids': text_ids.cpu()
        }
        del prompt_embeds, pooled_prompt_embeds, text_ids
        torch.cuda.empty_cache()

print(f"   已编码 {len(encoded_prompts_cache)} 个唯一prompts")

# 卸载encoder节省显存
print("🧹 卸载Text Encoders...")
del pipe.text_encoder, pipe.text_encoder_2, pipe.tokenizer, pipe.tokenizer_2
pipe.text_encoder = None
pipe.text_encoder_2 = None
pipe.tokenizer = None
pipe.tokenizer_2 = None
gc.collect()
torch.cuda.empty_cache()
pipe.to("cuda")

# ==========================================
# 4. 重建 FFT 密码本
# ==========================================
print("🔐 重建 FFT 密码本...")
secret_key = config['secret_key']
generator = torch.Generator(device='cuda').manual_seed(secret_key)
W_raw = torch.randn((1, 1024, 64), generator=generator, device='cuda', dtype=torch.float32)
W_spatial = W_raw.view(1, 32, 32, 64)

F_W = torch.fft.fftshift(torch.fft.fft2(W_spatial, dim=(1, 2)), dim=(1, 2))
h, w = 32, 32
Y, X = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
center_y, center_x = h // 2, w // 2
radius = torch.sqrt((Y - center_y)**2 + (X - center_x)**2).to('cuda')

r_inner = config['fft_radius_inner']
r_outer = config['fft_radius_outer']
mask_fft = ((radius >= r_inner) & (radius <= r_outer)).float().unsqueeze(0).unsqueeze(-1)

F_W_filtered = F_W * mask_fft
W_filtered = torch.fft.ifft2(torch.fft.ifftshift(F_W_filtered, dim=(1, 2)), dim=(1, 2)).real
W = W_filtered.view(1, 1024, 64).to(torch.bfloat16)
W = W / (torch.norm(W, dim=-1, keepdim=True) + 1e-8)

M_spatial = torch.zeros((1, 32, 32, 1), device='cuda', dtype=torch.bfloat16)
y_s, y_e, x_s, x_e = config['mask_region']
M_spatial[0, y_s:y_e, x_s:x_e, 0] = 1.0
M = M_spatial.view(1, 1024, 1)
W_anchored = W * M

# 核心掩码
mask_8x8 = np.zeros((8, 8))
mask_8x8[1:7, 1:7] = 1.0

# ==========================================
# 5. 提取签名函数
# ==========================================
def extract_signature(pil_image, prompt_embeds, pooled_prompt_embeds, text_ids):
    """从图像中提取签名"""
    img_tensor = T.ToTensor()(pil_image).unsqueeze(0).to("cuda", dtype=torch.bfloat16)
    img_tensor = (img_tensor - 0.5) * 2.0
    
    with torch.no_grad():
        z_0 = pipe.vae.encode(img_tensor).latent_dist.sample()
        z_0 = (z_0 - pipe.vae.config.shift_factor) * pipe.vae.config.scaling_factor
        z_0 = pipe._pack_latents(z_0, batch_size=1, num_channels_latents=16, height=64, width=64)
        
        img_ids = torch.zeros(32, 32, 3, device='cuda', dtype=torch.bfloat16)
        img_ids[..., 1] = torch.arange(32, device='cuda', dtype=torch.bfloat16).unsqueeze(1) / 31.0
        img_ids[..., 2] = torch.arange(32, device='cuda', dtype=torch.bfloat16).unsqueeze(0) / 31.0
        img_ids = img_ids.view(1, 1024, 3)
        
        v_pred = pipe.transformer(
            hidden_states=z_0,
            timestep=torch.tensor([pipe.scheduler.sigmas[-1]], device="cuda", dtype=torch.bfloat16) / 1000,
            pooled_projections=pooled_prompt_embeds,
            encoder_hidden_states=prompt_embeds,
            txt_ids=text_ids,
            img_ids=img_ids,
            return_dict=False,
        )[0]
    
    v_pred_spatial = v_pred.view(32, 32, 64)
    W_anchored_spatial = W_anchored.view(32, 32, 64)
    S = np.zeros((8, 8))
    
    for i in range(8):
        for j in range(8):
            W_patch = W_anchored_spatial[i*4:(i+1)*4, j*4:(j+1)*4, :].flatten().float()
            v_patch = v_pred_spatial[i*4:(i+1)*4, j*4:(j+1)*4, :].flatten().float()
            S[i, j] = torch.nn.functional.cosine_similarity(
                W_patch.unsqueeze(0), v_patch.unsqueeze(0)
            ).item()
    
    return S

def compare_signatures(S_orig, S_tamp, mask):
    """比较签名"""
    S_orig_masked = S_orig[mask == 1.0]
    S_tamp_masked = S_tamp[mask == 1.0]
    
    cos_sim = np.dot(S_orig_masked, S_tamp_masked) / (
        np.linalg.norm(S_orig_masked) * np.linalg.norm(S_tamp_masked) + 1e-8
    )
    
    mse = np.mean((S_orig_masked - S_tamp_masked) ** 2)
    mean_diff = abs(np.mean(S_orig_masked) - np.mean(S_tamp_masked))
    
    return float(cos_sim), float(mse), float(mean_diff)

# ==========================================
# 6. 篡改定位相关函数
# ==========================================
def compute_tamper_localization(S_orig, S_tamp, threshold_ratio=0.5):
    """
    计算篡改定位
    返回：
    - heatmap: 8x8 差分热力图
    - tamper_mask: 二值掩码，标识被篡改区域
    - detected_regions: 检测到的篡改区域列表
    """
    # 计算差分
    diff = S_orig - S_tamp
    
    # 创建热力图（基于差分的绝对值）
    heatmap = np.abs(diff)
    
    # 自适应阈值：使用核心区域的统计信息
    core_mask = mask_8x8 == 1.0
    core_diff = np.abs(diff[core_mask])
    
    # 设定阈值为核心区域差分的均值加上一定倍数的标准差
    if len(core_diff) > 0:
        threshold = np.mean(core_diff) + threshold_ratio * np.std(core_diff)
    else:
        threshold = np.mean(heatmap) + threshold_ratio * np.std(heatmap)
    
    # 二值化：找出显著偏离的区域
    tamper_mask = heatmap > threshold
    
    # 找出连通区域（简化版：直接返回mask）
    return heatmap, tamper_mask, threshold

def compute_iou(pred_mask, true_mask):
    """计算IoU（交并比）"""
    intersection = np.logical_and(pred_mask, true_mask).sum()
    union = np.logical_or(pred_mask, true_mask).sum()
    return intersection / (union + 1e-8)

def compute_f1_score(pred_mask, true_mask):
    """计算F1分数"""
    tp = np.logical_and(pred_mask, true_mask).sum()
    fp = np.logical_and(pred_mask, ~true_mask).sum()
    fn = np.logical_and(~pred_mask, true_mask).sum()
    
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    return f1, precision, recall

# ==========================================
# 7. 攻击函数
# ==========================================
def attack_jpeg(img, quality):
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=quality)
    return Image.open(buffer)

def attack_blur(img, sigma, kernel_size=9):
    tensor = T.ToTensor()(img).unsqueeze(0)
    blurred = F_vision.gaussian_blur(tensor, kernel_size=[kernel_size, kernel_size], sigma=[sigma, sigma])
    return T.ToPILImage()(blurred.squeeze(0))

def attack_black_block(img, block_size_ratio=0.25, position='center'):
    """
    在图像上添加黑色方块
    position: 'center', 'random', 'corner_tl', 'corner_tr', 'corner_bl', 'corner_br'
    """
    img_array = np.array(img)
    h, w = img_array.shape[:2]
    block_h = int(h * block_size_ratio)
    block_w = int(w * block_size_ratio)
    
    if position == 'center':
        y_start = (h - block_h) // 2
        x_start = (w - block_w) // 2
    elif position == 'random':
        y_start = np.random.randint(0, h - block_h)
        x_start = np.random.randint(0, w - block_w)
    elif position == 'corner_tl':
        y_start, x_start = 0, 0
    elif position == 'corner_tr':
        y_start, x_start = 0, w - block_w
    elif position == 'corner_bl':
        y_start, x_start = h - block_h, 0
    elif position == 'corner_br':
        y_start, x_start = h - block_h, w - block_w
    else:
        y_start = (h - block_h) // 2
        x_start = (w - block_w) // 2
    
    attacked = img_array.copy()
    attacked[y_start:y_start+block_h, x_start:x_start+block_w] = 0
    
    # 返回攻击后的图和真实篡改掩码（8x8网格）
    true_mask = np.zeros((8, 8), dtype=bool)
    y_start_8 = int(y_start / h * 8)
    y_end_8 = int((y_start + block_h) / h * 8)
    x_start_8 = int(x_start / w * 8)
    x_end_8 = int((x_start + block_w) / w * 8)
    
    y_start_8 = max(0, min(7, y_start_8))
    y_end_8 = max(0, min(8, y_end_8))
    x_start_8 = max(0, min(7, x_start_8))
    x_end_8 = max(0, min(8, x_end_8))
    
    true_mask[y_start_8:y_end_8, x_start_8:x_end_8] = True
    
    return Image.fromarray(attacked), true_mask, (x_start, y_start, block_w, block_h)

def attack_multiple_blocks(img, num_blocks=3, block_size_ratio=0.15):
    """多个随机黑色方块"""
    img_array = np.array(img)
    h, w = img_array.shape[:2]
    true_mask = np.zeros((8, 8), dtype=bool)
    
    for _ in range(num_blocks):
        block_h = int(h * block_size_ratio)
        block_w = int(w * block_size_ratio)
        y_start = np.random.randint(0, max(1, h - block_h))
        x_start = np.random.randint(0, max(1, w - block_w))
        
        img_array[y_start:y_start+block_h, x_start:x_start+block_w] = 0
        
        # 更新mask
        y_start_8 = int(y_start / h * 8)
        y_end_8 = int((y_start + block_h) / h * 8)
        x_start_8 = int(x_start / w * 8)
        x_end_8 = int((x_start + block_w) / w * 8)
        
        y_start_8 = max(0, min(7, y_start_8))
        y_end_8 = max(0, min(8, y_end_8))
        x_start_8 = max(0, min(7, x_start_8))
        x_end_8 = max(0, min(8, x_end_8))
        
        true_mask[y_start_8:y_end_8, x_start_8:x_end_8] = True
    
    return Image.fromarray(img_array), true_mask

# ==========================================
# 8. 可视化函数
# ==========================================
def visualize_tamper_detection(img_orig, img_attacked, S_orig, S_tamp, true_mask, 
                                save_path, attack_name, metrics=None):
    """
    创建篡改检测可视化对比图
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 原图
    axes[0, 0].imshow(img_orig)
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    # 攻击后图像
    axes[0, 1].imshow(img_attacked)
    axes[0, 1].set_title(f'Attacked: {attack_name}')
    axes[0, 1].axis('off')
    
    # 差分热力图
    heatmap, pred_mask, threshold = compute_tamper_localization(S_orig, S_tamp)
    im = axes[0, 2].imshow(heatmap, cmap='hot', interpolation='nearest')
    axes[0, 2].set_title(f'Differential Heatmap\n(threshold={threshold:.3f})')
    axes[0, 2].axis('off')
    plt.colorbar(im, ax=axes[0, 2])
    
    # 真实篡改区域
    axes[1, 0].imshow(true_mask, cmap='Reds', interpolation='nearest')
    axes[1, 0].set_title('True Tampered Regions')
    axes[1, 0].axis('off')
    
    # 检测到的篡改区域
    axes[1, 1].imshow(pred_mask, cmap='Blues', interpolation='nearest')
    axes[1, 1].set_title('Detected Tampered Regions')
    axes[1, 1].axis('off')
    
    # 对比图（绿色=正确检测，红色=漏检，黄色=误检）
    comparison = np.zeros((*pred_mask.shape, 3))
    tp = np.logical_and(pred_mask, true_mask)
    fp = np.logical_and(pred_mask, ~true_mask)
    fn = np.logical_and(~pred_mask, true_mask)
    
    comparison[tp] = [0, 1, 0]  # 绿色：正确
    comparison[fp] = [1, 1, 0]  # 黄色：误检
    comparison[fn] = [1, 0, 0]  # 红色：漏检
    
    axes[1, 2].imshow(comparison, interpolation='nearest')
    title = 'Detection Comparison\n(Green=TP, Yellow=FP, Red=FN)'
    if metrics:
        title += f"\nIoU={metrics['iou']:.3f}, F1={metrics['f1']:.3f}"
    axes[1, 2].set_title(title)
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return heatmap, pred_mask

# ==========================================
# 9. 创建输出目录
# ==========================================
tamper_viz_dir = os.path.join(output_dir, 'tamper_visualizations')
os.makedirs(tamper_viz_dir, exist_ok=True)

# ==========================================
# 10. 主测试循环
# ==========================================
print(f"\n🔬 开始鲁棒性测试与篡改定位评估...")

# 定义测试的攻击（精简版，重点加篡改攻击）
test_attacks = [
    # 传统鲁棒性攻击
    {'name': 'clean', 'type': 'baseline', 'params': {}};
    {'name': 'jpeg_50', 'type': 'jpeg', 'params': {'quality': 50}},
    {'name': 'blur_1.0', 'type': 'blur', 'params': {'sigma': 1.0, 'kernel_size': 5}},
    
    # 篡改攻击（关键新增）
    {'name': 'black_block_center', 'type': 'black_block', 'params': {'block_size_ratio': 0.25, 'position': 'center'}},
    {'name': 'black_block_random', 'type': 'black_block', 'params': {'block_size_ratio': 0.25, 'position': 'random'}},
    {'name': 'black_block_corner', 'type': 'black_block', 'params': {'block_size_ratio': 0.2, 'position': 'corner_tl'}},
    {'name': 'multiple_blocks', 'type': 'multiple_blocks', 'params': {'num_blocks': 3, 'block_size_ratio': 0.15}},
]

results = {
    'experiment_name': experiment_name,
    'timestamp': datetime.now().isoformat(),
    'config': config,
    'robustness': {},
    'tamper_localization': []
}

# 记录每种攻击的统计
for attack in test_attacks:
    results['robustness'][attack['name']] = {
        'scores': [],
        'mses': [],
        'mean_diffs': []
    }

# 遍历每张图像
for img_idx, img_info in enumerate(tqdm(images_info[:5], desc="图像进度")):  # 只测前5张节省时间
    img_id = img_info['id']
    prompt_text = img_info['prompt']
    
    # 加载原图和签名
    img_path = os.path.join(output_dir, img_info['image_file'])
    sig_path = os.path.join(output_dir, img_info['signature_file'])
    
    img = Image.open(img_path).convert("RGB")
    S_orig = np.load(sig_path)
    
    # 使用预编码的prompts
    encoded = encoded_prompts_cache[prompt_text]
    prompt_embeds = encoded['prompt_embeds'].to("cuda")
    pooled_prompt_embeds = encoded['pooled_prompt_embeds'].to("cuda")
    text_ids = encoded['text_ids'].to("cuda")
    
    img_result = {
        'id': img_id,
        'attacks': {}
    }
    
    # 对每种攻击进行测试
    for attack_def in test_attacks:
        attack_name = attack_def['name']
        attack_type = attack_def['type']
        params = attack_def['params']
        
        attack_scores = []
        attack_ious = []
        attack_f1s = []
        
        for run in range(num_runs):
            # 执行攻击
            if attack_type == 'baseline':
                attacked_img = img
                true_mask = np.zeros((8, 8), dtype=bool)
            elif attack_type == 'jpeg':
                attacked_img = attack_jpeg(img, params['quality'])
                true_mask = np.zeros((8, 8), dtype=bool)
            elif attack_type == 'blur':
                tensor = T.ToTensor()(img).unsqueeze(0)
                blurred = F_vision.gaussian_blur(tensor, kernel_size=[params['kernel_size']]*2, 
                                                  sigma=[params['sigma']]*2)
                attacked_img = T.ToPILImage()(blurred.squeeze(0))
                true_mask = np.zeros((8, 8), dtype=bool)
            elif attack_type == 'black_block':
                attacked_img, true_mask, (bx, by, bw, bh) = attack_black_block(
                    img, params['block_size_ratio'], params['position']
                )
            elif attack_type == 'multiple_blocks':
                attacked_img, true_mask = attack_multiple_blocks(
                    img, params['num_blocks'], params['block_size_ratio']
                )
            
            # 提取签名
            S_tamp = extract_signature(attacked_img, prompt_embeds, pooled_prompt_embeds, text_ids)
            cos_sim, mse, mean_diff = compare_signatures(S_orig, S_tamp, mask_8x8)
            
            attack_scores.append(cos_sim)
            results['robustness'][attack_name]['scores'].append(cos_sim)
            results['robustness'][attack_name]['mses'].append(mse)
            results['robustness'][attack_name]['mean_diffs'].append(mean_diff)
            
            # 篡改定位评估（仅对篡改攻击）
            if attack_type in ['black_block', 'multiple_blocks']:
                heatmap, pred_mask, threshold = compute_tamper_localization(S_orig, S_tamp)
                iou = compute_iou(pred_mask, true_mask)
                f1, precision, recall = compute_f1_score(pred_mask, true_mask)
                
                attack_ious.append(iou)
                attack_f1s.append(f1)
                
                # 第一张图，第一次运行，保存可视化
                if img_idx == 0 and run == 0:
                    viz_path = os.path.join(tamper_viz_dir, f'{img_id}_{attack_name}_detection.png')
                    metrics = {'iou': iou, 'f1': f1, 'precision': precision, 'recall': recall}
                    visualize_tamper_detection(img, attacked_img, S_orig, S_tamp, 
                                              true_mask, viz_path, attack_name, metrics)
                    print(f"   保存可视化: {viz_path}")
        
        # 计算攻击的统计
        img_result['attacks'][attack_name] = {
            'mean_score': float(np.mean(attack_scores)),
            'std_score': float(np.std(attack_scores)),
        }
        
        if attack_ious:
            img_result['attacks'][attack_name]['mean_iou'] = float(np.mean(attack_ious))
            img_result['attacks'][attack_name]['mean_f1'] = float(np.mean(attack_f1s))
            
            results['tamper_localization'].append({
                'image_id': img_id,
                'attack': attack_name,
                'iou': float(np.mean(attack_ious)),
                'f1': float(np.mean(attack_f1s)),
                'score_drop': float(results['robustness']['clean']['scores'][-1] - np.mean(attack_scores))
            })
    
    results[f'image_{img_id}'] = img_result
    
    # 清理
    del prompt_embeds, pooled_prompt_embeds, text_ids
    torch.cuda.empty_cache()

# ==========================================
# 11. 计算总体统计
# ==========================================
print("\n📊 计算总体统计...")

for attack_name in results['robustness']:
    scores = results['robustness'][attack_name]['scores']
    results['robustness'][attack_name]['overall'] = {
        'mean': float(np.mean(scores)),
        'std': float(np.std(scores)),
        'median': float(np.median(scores)),
        'min': float(np.min(scores)),
        'max': float(np.max(scores))
    }

# 篡改定位总体统计
if results['tamper_localization']:
    all_ious = [t['iou'] for t in results['tamper_localization']]
    all_f1s = [t['f1'] for t in results['tamper_localization']]
    results['tamper_localization_summary'] = {
        'mean_iou': float(np.mean(all_ious)),
        'mean_f1': float(np.mean(all_f1s)),
        'std_iou': float(np.std(all_ious)),
        'std_f1': float(np.std(all_f1s))
    }

# ==========================================
# 12. 保存结果
# ==========================================
result_path = os.path.join(output_dir, f'{experiment_name}_tamper_test_results.json')
with open(result_path, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n✅ 测试完成！结果保存至: {result_path}")

# ==========================================
# 13. 打印报告
# ==========================================
print("\n" + "="*80)
print("📊 鲁棒性测试汇总")
print("="*80)
print(f"{'攻击类型':<25} {'平均相似度':>12} {'标准差':>10} {'保留率':>10} {'状态':>8}")
print("-"*80)

baseline = results['robustness']['clean']['overall']['mean']
for attack_name in ['clean', 'jpeg_50', 'blur_1.0', 'black_block_center', 'black_block_random', 'multiple_blocks']:
    if attack_name in results['robustness']:
        stats = results['robustness'][attack_name]['overall']
        retention = stats['mean'] / baseline * 100 if attack_name != 'clean' else 100.0
        
        if stats['mean'] > 0.85:
            status = "✅ 强"
        elif stats['mean'] > 0.70:
            status = "🟢 良"
        elif stats['mean'] > 0.50:
            status = "🟡 中"
        elif stats['mean'] > 0.30:
            status = "🟠 弱"
        else:
            status = "🔴 差"
        
        print(f"{attack_name:<25} {stats['mean']:>12.4f} {stats['std']:>10.4f} {retention:>9.1f}% {status:>8}")

print("="*80)

# 篡改定位报告
if 'tamper_localization_summary' in results:
    print("\n" + "="*80)
    print("🎯 篡改定位性能评估")
    print("="*80)
    summary = results['tamper_localization_summary']
    print(f"平均 IoU (交并比): {summary['mean_iou']:.4f} ± {summary['std_iou']:.4f}")
    print(f"平均 F1 分数:      {summary['mean_f1']:.4f} ± {summary['std_f1']:.4f}")
    print("")
    print("IoU 解释:")
    print("  > 0.70: 优秀 (精确定位篡改区域)")
    print("  0.50-0.70: 良好 (大致定位正确)")
    print("  0.30-0.50: 中等 (部分重叠)")
    print("  < 0.30: 较差 (定位失败)")
    print("="*80)

print(f"\n🖼️ 可视化结果保存在: {tamper_viz_dir}")
