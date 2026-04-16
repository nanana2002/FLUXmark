#!/usr/bin/env python3
"""
OrthoFlow 规模化鲁棒性测试脚本
对生成的图像进行多种攻击测试，每种攻击运行多次，统计结果
"""

import torch
from diffusers import FluxPipeline
import torchvision.transforms as T
from torchvision.transforms import functional as F_vision
from PIL import Image
import numpy as np
import io
import os
import gc
import json
from datetime import datetime
from tqdm import tqdm

os.environ['HF_HOME'] = '/data/daiyina/hf_cache'
output_dir = '/data/daiyina/project_flux/pic'

# ==========================================
# 1. 加载实验配置
# ==========================================
print("📋 加载实验配置...")
with open('prompts.json', 'r') as f:
    config = json.load(f)

experiment_name = config['experiment_name']
num_runs = 5  # 每种攻击运行次数

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
# 4. 定义攻击和提取函数
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

# 定义攻击函数
def attack_jpeg(img, quality):
    """JPEG压缩攻击"""
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=quality)
    return Image.open(buffer)

def attack_blur(img, sigma, kernel_size=9):
    """高斯模糊攻击"""
    tensor = T.ToTensor()(img).unsqueeze(0)
    blurred = F_vision.gaussian_blur(tensor, kernel_size=[kernel_size, kernel_size], sigma=[sigma, sigma])
    return T.ToPILImage()(blurred.squeeze(0))

def attack_crop_center(img, crop_ratio=0.5):
    """中心裁剪攻击"""
    w, h = img.size
    left = int(w * (1 - crop_ratio) / 2)
    top = int(h * (1 - crop_ratio) / 2)
    right = w - left
    bottom = h - top
    cropped = img.crop((left, top, right, bottom))
    return cropped.resize((512, 512), Image.Resampling.LANCZOS)

def attack_noise(img, std=0.05):
    """加噪声攻击"""
    tensor = T.ToTensor()(img)
    noise = torch.randn_like(tensor) * std
    noisy = torch.clamp(tensor + noise, 0, 1)
    return T.ToPILImage()(noisy)

def attack_resize(img, scale=0.5):
    """缩放攻击"""
    w, h = img.size
    small = img.resize((int(w*scale), int(h*scale)), Image.Resampling.LANCZOS)
    return small.resize((w, h), Image.Resampling.LANCZOS)



# ==========================================
# 4. 保存攻击示例图像
# ==========================================
save_attack_examples = True
attack_examples_dir = os.path.join(output_dir, 'attack_examples')
if save_attack_examples:
    os.makedirs(attack_examples_dir, exist_ok=True)
    print(f"\n🖼️ 攻击示例将保存至: {attack_examples_dir}")

# ==========================================
# 5. 主测试循环
# ==========================================
print(f"\n🔬 开始鲁棒性测试...")

test_results = {
    'experiment_name': experiment_name,
    'timestamp': datetime.now().isoformat(),
    'num_runs_per_attack': num_runs,
    'config': config,
    'attacks': [
        {'name': 'clean', 'type': 'baseline', 'params': {}},
        {'name': 'jpeg_75', 'type': 'jpeg', 'params': {'quality': 75}},
        {'name': 'jpeg_50', 'type': 'jpeg', 'params': {'quality': 50}},
        {'name': 'jpeg_30', 'type': 'jpeg', 'params': {'quality': 30}},
        {'name': 'jpeg_20', 'type': 'jpeg', 'params': {'quality': 20}},
        {'name': 'blur_0.5', 'type': 'blur', 'params': {'sigma': 0.5, 'kernel_size': 5}},
        {'name': 'blur_1.0', 'type': 'blur', 'params': {'sigma': 1.0, 'kernel_size': 5}},
        {'name': 'blur_2.0', 'type': 'blur', 'params': {'sigma': 2.0, 'kernel_size': 9}},
        {'name': 'blur_3.0', 'type': 'blur', 'params': {'sigma': 3.0, 'kernel_size': 11}},
        {'name': 'crop_0.75', 'type': 'crop', 'params': {'crop_ratio': 0.75}},
        {'name': 'crop_0.50', 'type': 'crop', 'params': {'crop_ratio': 0.50}},
        {'name': 'noise_0.03', 'type': 'noise', 'params': {'std': 0.03}},
        {'name': 'noise_0.05', 'type': 'noise', 'params': {'std': 0.05}},
        {'name': 'resize_0.75', 'type': 'resize', 'params': {'scale': 0.75}},
        {'name': 'resize_0.50', 'type': 'resize', 'params': {'scale': 0.50}},
    ],
    'results': []
}

for img_idx, img_info in enumerate(tqdm(images_info, desc="图像进度")):
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
        'prompt': prompt_text,
        'attacks': {}
    }
    
    # 对每种攻击进行测试
    for attack_def in test_results['attacks']:
        attack_name = attack_def['name']
        attack_type = attack_def['type']
        params = attack_def['params']
        
        scores = []
        mses = []
        mean_diffs = []
        
        # 保存攻击示例（仅第一张图）
        example_saved = False
        
        # 多次运行
        for run in range(num_runs):
            # 执行攻击
            if attack_type == 'baseline':
                attacked_img = img
            elif attack_type == 'jpeg':
                attacked_img = attack_jpeg(img, params['quality'])
            elif attack_type == 'blur':
                attacked_img = attack_blur(img, params['sigma'], params.get('kernel_size', 9))
            elif attack_type == 'crop':
                attacked_img = attack_crop_center(img, params['crop_ratio'])
            elif attack_type == 'noise':
                attacked_img = attack_noise(img, params['std'])
            elif attack_type == 'resize':
                attacked_img = attack_resize(img, params['scale'])
            
            # 保存攻击示例（仅第一张图，第一次运行）
            if img_idx == 0 and run == 0 and attack_name != 'clean' and not example_saved and save_attack_examples:
                if attack_type == 'jpeg':
                    ext = 'jpg'
                else:
                    ext = 'png'
                example_path = os.path.join(attack_examples_dir, f'example_{attack_name}.{ext}')
                attacked_img.save(example_path)
                example_saved = True
            
            # 提取签名并比较
            S_tamp = extract_signature(attacked_img, prompt_embeds, pooled_prompt_embeds, text_ids)
            cos_sim, mse, mean_diff = compare_signatures(S_orig, S_tamp, mask_8x8)
            
            scores.append(cos_sim)
            mses.append(mse)
            mean_diffs.append(mean_diff)
        
        # 统计结果
        img_result['attacks'][attack_name] = {
            'cos_sim_mean': float(np.mean(scores)),
            'cos_sim_std': float(np.std(scores)),
            'cos_sim_min': float(np.min(scores)),
            'cos_sim_max': float(np.max(scores)),
            'mse_mean': float(np.mean(mses)),
            'mean_diff_mean': float(np.mean(mean_diffs)),
            'retention_rate': float(np.mean(scores) / img_result['attacks'].get('clean', {}).get('cos_sim_mean', np.mean(scores)) * 100) if attack_name != 'clean' else 100.0
        }
    
    test_results['results'].append(img_result)
    
    # 清理
    del prompt_embeds, pooled_prompt_embeds, text_ids
    torch.cuda.empty_cache()

# ==========================================
# 6. 计算总体统计
# ==========================================
print("\n📊 计算总体统计...")

overall_stats = {}
for attack_def in test_results['attacks']:
    attack_name = attack_def['name']
    all_scores = [r['attacks'][attack_name]['cos_sim_mean'] for r in test_results['results']]
    
    overall_stats[attack_name] = {
        'mean': float(np.mean(all_scores)),
        'std': float(np.std(all_scores)),
        'median': float(np.median(all_scores)),
        'min': float(np.min(all_scores)),
        'max': float(np.max(all_scores)),
        'q25': float(np.percentile(all_scores, 25)),
        'q75': float(np.percentile(all_scores, 75))
    }

test_results['overall_stats'] = overall_stats

# ==========================================
# 7. 保存结果
# ==========================================
result_path = os.path.join(output_dir, f'{experiment_name}_test_results.json')
with open(result_path, 'w') as f:
    json.dump(test_results, f, indent=2)

print(f"\n✅ 测试完成！结果保存至: {result_path}")

# ==========================================
# 8. 打印汇总表格
# ==========================================
print("\n" + "="*80)
print("📊 鲁棒性测试汇总结果 (20张图像 × 5次运行)")
print("="*80)
print(f"{'攻击类型':<20} {'平均相似度':>12} {'标准差':>10} {'中位数':>12} {'保留率':>10} {'状态':>8}")
print("-"*80)

baseline_mean = overall_stats['clean']['mean']

for attack_def in test_results['attacks']:
    name = attack_def['name']
    stats = overall_stats[name]
    retention = stats['mean'] / baseline_mean * 100 if name != 'clean' else 100.0
    
    if stats['mean'] > 0.85:
        status = "✅ 强"
    elif stats['mean'] > 0.70:
        status = "🟢 良"
    elif stats['mean'] > 0.55:
        status = "🟡 中"
    elif stats['mean'] > 0.40:
        status = "🟠 弱"
    else:
        status = "🔴 差"
    
    print(f"{name:<20} {stats['mean']:>12.4f} {stats['std']:>10.4f} {stats['median']:>12.4f} {retention:>9.1f}% {status:>8}")

print("="*80)

# ==========================================
# 9. 打印论文用表格数据
# ==========================================
print("\n📄 论文 Table 数据 (LaTeX 格式):")
print("-"*80)
print("Attack Type & Mean & Std & Retention \\\\")
print("\\hline")
for attack_def in test_results['attacks']:
    name = attack_def['name'].replace('_', '\\_')
    stats = overall_stats[attack_def['name']]
    retention = stats['mean'] / baseline_mean * 100
    print(f"{name} & {stats['mean']:.4f} & {stats['std']:.4f} & {retention:.1f}\\% \\\\")
