#!/usr/bin/env python3
"""
批量提取所有攻击图像的水印签名（内存优化版 - 22GB显存适配）
提取完成后保存到 watermark_extra/ 目录，供分析脚本使用

使用方法:
    python3 extract_watermarks.py
"""

import torch
from diffusers import FluxPipeline
import torchvision.transforms as T
from PIL import Image
import numpy as np
import os
import gc
import json
from datetime import datetime
from tqdm import tqdm

# 加载配置
with open('config.json', 'r') as f:
    config = json.load(f)

os.environ['HF_HOME'] = config['hf_cache']

# 路径设置
base_output_dir = os.path.join(config['output_base_dir'], 'pic')
watermark_extra_dir = os.path.join(config['output_base_dir'], 'watermark_extra')
os.makedirs(watermark_extra_dir, exist_ok=True)

watermarked_dir = os.path.join(base_output_dir, 'watermarked_img')
attack_dir = os.path.join(base_output_dir, 'attack_watermarked_img')
no_watermark_dir = os.path.join(base_output_dir, 'no_watermarked_img')

# 加载摘要
print("📋 加载实验数据...")
with open(os.path.join(watermarked_dir, f'{config["experiment_name"]}_watermark_summary.json'), 'r') as f:
    watermark_summary = json.load(f)

attack_summary_path = os.path.join(attack_dir, f'{config["experiment_name"]}_attack_summary.json')
if os.path.exists(attack_summary_path):
    with open(attack_summary_path, 'r') as f:
        attack_summary = json.load(f)
    attack_configs = attack_summary.get('attacks', [])
else:
    print("❌ 未找到攻击摘要文件，请先运行 attack_img_generate.py")
    exit(1)

images_info = watermark_summary['images']
print(f"   水印图像: {len(images_info)} 张")
print(f"   攻击类型: {len(attack_configs)} 种")

# ==========================================
# 1. 加载模型（关键：先不移动到CUDA）
# ==========================================
print("\n🚀 加载 FLUX 模型...")
pipe = FluxPipeline.from_pretrained(
    config['model_path'],
    torch_dtype=torch.bfloat16
)

# ==========================================
# 2. 预编码所有prompts
# ==========================================
print("🔤 预编码所有prompts...")
encoded_prompts_cache = {}
unique_prompts = set()

for img_info in images_info:
    unique_prompts.add(img_info['prompt'])

for prompt_text in tqdm(unique_prompts, desc="编码进度"):
    with torch.no_grad():
        prompt_embeds, pooled_prompt_embeds, text_ids = pipe.encode_prompt(
            prompt=prompt_text, prompt_2=None, max_sequence_length=256
        )
    # 关键：移到CPU保存
    encoded_prompts_cache[prompt_text] = {
        'prompt_embeds': prompt_embeds.cpu(),
        'pooled_prompt_embeds': pooled_prompt_embeds.cpu(),
        'text_ids': text_ids.cpu()
    }
    del prompt_embeds, pooled_prompt_embeds, text_ids
    torch.cuda.empty_cache()

print(f"   已编码 {len(encoded_prompts_cache)} 个唯一prompts")

# ==========================================
# 3. 卸载Text Encoder节省显存
# ==========================================
print("🧹 卸载Text Encoders...")
del pipe.text_encoder, pipe.text_encoder_2, pipe.tokenizer, pipe.tokenizer_2
pipe.text_encoder = None
pipe.text_encoder_2 = None
pipe.tokenizer = None
pipe.tokenizer_2 = None
gc.collect()
torch.cuda.empty_cache()
pipe.to("cuda")
print("   模型已加载到GPU")

# ==========================================
# 4. 重建FFT密码本
# ==========================================
print("🔐 重建FFT密码本...")
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

mask_8x8 = np.zeros((8, 8))
mask_8x8[1:7, 1:7] = 1.0

# ==========================================
# 5. 提取签名函数
# ==========================================
def extract_signature(pil_image, prompt_embeds, pooled_prompt_embeds, text_ids):
    """从图像中提取签名（GPU版本）"""
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

# ==========================================
# 6. 提取所有签名
# ==========================================
print("\n" + "="*60)
print("开始提取签名...")
print("="*60)

extraction_results = {
    'experiment_name': config['experiment_name'],
    'timestamp': datetime.now().isoformat(),
    'signatures': {}
}

# 6.1 提取原始水印图像的签名 S_orig
print("\n1. 提取原始水印图像签名 (S_orig)...")
for img_info in tqdm(images_info, desc="原始图像"):
    img_id = img_info['id']
    prompt = img_info['prompt']

    img_path = os.path.join(watermarked_dir, img_info['image_file'])
    S_orig_path = os.path.join(watermark_extra_dir, f"{img_id}_S_orig.npy")

    # 检查是否已提取
    if os.path.exists(S_orig_path):
        continue

    img = Image.open(img_path).convert('RGB')
    encoded = encoded_prompts_cache[prompt]

    # 关键：需要时移到GPU，用完立即释放
    prompt_embeds = encoded['prompt_embeds'].to("cuda")
    pooled_prompt_embeds = encoded['pooled_prompt_embeds'].to("cuda")
    text_ids = encoded['text_ids'].to("cuda")

    S_orig = extract_signature(img, prompt_embeds, pooled_prompt_embeds, text_ids)
    np.save(S_orig_path, S_orig)

    del prompt_embeds, pooled_prompt_embeds, text_ids
    torch.cuda.empty_cache()

# 6.2 提取无水印图像的签名（负样本）
print("\n2. 提取无水印图像签名 (负样本)...")
no_wm_summary_path = os.path.join(no_watermark_dir, f'{config["experiment_name"]}_no_watermark_summary.json')
if os.path.exists(no_wm_summary_path):
    with open(no_wm_summary_path, 'r') as f:
        no_wm_summary = json.load(f)

    for img_info in tqdm(no_wm_summary['images'], desc="无水印图像"):
        img_id = img_info['id']
        prompt = img_info['prompt']

        img_path = os.path.join(no_watermark_dir, img_info['image_file'])
        S_no_wm_path = os.path.join(watermark_extra_dir, f"{img_id}_S_no_wm.npy")

        if os.path.exists(S_no_wm_path):
            continue

        img = Image.open(img_path).convert('RGB')
        encoded = encoded_prompts_cache.get(prompt)
        if encoded is None:
            # 使用第一个prompt作为fallback
            encoded = encoded_prompts_cache.get(watermark_summary['images'][0]['prompt'])

        prompt_embeds = encoded['prompt_embeds'].to("cuda")
        pooled_prompt_embeds = encoded['pooled_prompt_embeds'].to("cuda")
        text_ids = encoded['text_ids'].to("cuda")

        S_no_wm = extract_signature(img, prompt_embeds, pooled_prompt_embeds, text_ids)
        np.save(S_no_wm_path, S_no_wm)

        del prompt_embeds, pooled_prompt_embeds, text_ids
        torch.cuda.empty_cache()

# 6.3 提取攻击后图像的签名 S_tamp
print("\n3. 提取攻击后图像签名 (S_tamp)...")
for attack_name in tqdm(attack_configs, desc="攻击类型"):
    attack_results = attack_summary.get('images', [])

    for img_info in attack_results:
        img_id = img_info['id']
        prompt = img_info['prompt']

        # 检查该攻击是否存在
        if attack_name not in img_info.get('attacks', {}):
            continue

        # 构建攻击图像路径
        attacked_img_name = img_info['attacks'][attack_name].get('image_file', f"{img_id}.png")
        attacked_img_path = os.path.join(attack_dir, attack_name, attacked_img_name)

        if not os.path.exists(attacked_img_path):
            continue

        # 检查是否已提取
        S_tamp_path = os.path.join(watermark_extra_dir, f"{img_id}_{attack_name}_S_tamp.npy")
        if os.path.exists(S_tamp_path):
            continue

        try:
            img = Image.open(attacked_img_path).convert('RGB')
            encoded = encoded_prompts_cache[prompt]

            prompt_embeds = encoded['prompt_embeds'].to("cuda")
            pooled_prompt_embeds = encoded['pooled_prompt_embeds'].to("cuda")
            text_ids = encoded['text_ids'].to("cuda")

            S_tamp = extract_signature(img, prompt_embeds, pooled_prompt_embeds, text_ids)
            np.save(S_tamp_path, S_tamp)

            del prompt_embeds, pooled_prompt_embeds, text_ids
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"\n   ❌ {img_id} {attack_name} 失败: {e}")
            continue

# ==========================================
# 7. 保存提取摘要
# ==========================================
summary_path = os.path.join(watermark_extra_dir, 'extraction_summary.json')
with open(summary_path, 'w') as f:
    json.dump({
        'experiment_name': config['experiment_name'],
        'timestamp': datetime.now().isoformat(),
        'config': config,
        'num_images': len(images_info),
        'num_attacks': len(attack_configs),
        'attack_types': attack_configs,
        'output_dir': watermark_extra_dir
    }, f, indent=2)

# 清理
del pipe, W, W_anchored
gc.collect()
torch.cuda.empty_cache()

print(f"\n" + "="*60)
print("✅ 签名提取完成！")
print("="*60)
print(f"   签名保存位置: {watermark_extra_dir}")
print(f"   提取摘要: {summary_path}")
print(f"\n   提取的签名文件:")
print(f"   - S_orig: {len([f for f in os.listdir(watermark_extra_dir) if f.endswith('_S_orig.npy')])} 个")
print(f"   - S_no_wm: {len([f for f in os.listdir(watermark_extra_dir) if f.endswith('_S_no_wm.npy')])} 个")
print(f"   - S_tamp: {len([f for f in os.listdir(watermark_extra_dir) if f.endswith('_S_tamp.npy')])} 个")
print(f"\n   现在可以运行: python3 analyze_robustness_result.py")
