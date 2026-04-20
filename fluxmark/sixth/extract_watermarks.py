#!/usr/bin/env python3
"""
批量提取所有攻击图像的水印签名（极致性能版 - 50GB显存全速运行）
优化策略：
- 保持所有模型常驻GPU
- 预编码所有prompts并缓存于GPU
- 批量提取签名（batch processing）
- 支持 --only-s-tamp 参数仅补提缺失的攻击签名
"""

import os
import sys
import json

# 先读取配置设置 GPU（必须在 import torch 之前）
with open('config.json', 'r') as f:
    config = json.load(f)
os.environ['CUDA_VISIBLE_DEVICES'] = str(config.get('gpu_id', 0))
os.environ['HF_HOME'] = config['hf_cache']

import torch
from diffusers import FluxPipeline
import torchvision.transforms as T
from PIL import Image
import numpy as np
from datetime import datetime
from tqdm import tqdm

only_s_tamp = ('--only-s-tamp' in sys.argv)

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
if only_s_tamp:
    print("   模式: 仅补提缺失的 S_tamp")

# ==========================================
# 1. 加载模型（直接到GPU，保持常驻）
# ==========================================
print("\n🚀 加载 FLUX 模型到GPU...")
pipe = FluxPipeline.from_pretrained(
    config['model_path'],
    torch_dtype=torch.bfloat16
)

pipe.enable_attention_slicing(slice_size="auto")
pipe.to("cuda")
print(f"   模型已加载到GPU，显存使用: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

# ==========================================
# 2. 预编码所有prompts（缓存于GPU）
# ==========================================
print("🔤 预编码所有prompts到GPU缓存...")
encoded_prompts_cache = {}
unique_prompts = set()

for img_info in images_info:
    unique_prompts.add(img_info['prompt'])

no_wm_summary_path = os.path.join(no_watermark_dir, f'{config["experiment_name"]}_no_watermark_summary.json')
if os.path.exists(no_wm_summary_path):
    with open(no_wm_summary_path, 'r') as f:
        no_wm_summary = json.load(f)
    for img_info in no_wm_summary['images']:
        unique_prompts.add(img_info['prompt'])

for prompt_text in tqdm(unique_prompts, desc="编码进度"):
    with torch.no_grad():
        prompt_embeds, pooled_prompt_embeds, text_ids = pipe.encode_prompt(
            prompt=prompt_text, prompt_2=None, max_sequence_length=256
        )
    encoded_prompts_cache[prompt_text] = {
        'prompt_embeds': prompt_embeds,
        'pooled_prompt_embeds': pooled_prompt_embeds,
        'text_ids': text_ids
    }

print(f"   已编码 {len(encoded_prompts_cache)} 个唯一prompts到GPU")
print(f"   当前显存使用: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

# 预编码空 prompt（盲提取模式缓存，只编码一次）
print("🔤 预编码空 prompt（盲提取模式缓存）...")
with torch.no_grad():
    empty_prompt_embeds, empty_pooled_prompt_embeds, empty_text_ids = pipe.encode_prompt(
        prompt="", prompt_2=None, max_sequence_length=256
    )
empty_prompt_cache = {
    'prompt_embeds': empty_prompt_embeds,
    'pooled_prompt_embeds': empty_pooled_prompt_embeds,
    'text_ids': empty_text_ids
}
print("   空 prompt 编码已缓存")

# ==========================================
# 3. 重建FFT密码本（常驻GPU）
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

print(f"   密码本已重建并常驻GPU（全图统一提取）")

# ==========================================
# 4. 提取签名函数（批量版本）
# ==========================================
def extract_signature_batch(pil_images, prompt_embeds_list, pooled_embeds_list, text_ids_list, grid_size=32, use_empty_prompt=False):
    """批量从图像中提取签名（高性能GPU版本，支持可变网格尺寸）
    
    Args:
        use_empty_prompt: 若为 True，使用空字符串 "" 的 prompt embedding 进行盲提取
    """
    batch_size = len(pil_images)
    patch_size = 32 // grid_size

    # 批量图像预处理（统一 resize 到 512x512）
    img_tensors = []
    for img in pil_images:
        img = img.resize((512, 512), Image.LANCZOS)
        img_tensor = T.ToTensor()(img).unsqueeze(0)
        img_tensors.append(img_tensor)
    img_batch = torch.cat(img_tensors).to("cuda", dtype=torch.bfloat16)
    img_batch = (img_batch - 0.5) * 2.0

    with torch.no_grad():
        # 批量VAE编码
        z_0_list = []
        for img_tensor in img_batch:
            z_0 = pipe.vae.encode(img_tensor.unsqueeze(0)).latent_dist.sample()
            z_0 = (z_0 - pipe.vae.config.shift_factor) * pipe.vae.config.scaling_factor
            z_0 = pipe._pack_latents(z_0, batch_size=1, num_channels_latents=16, height=64, width=64)
            z_0_list.append(z_0)

        # 批量构造img_ids
        h = w = 32
        img_ids = torch.zeros(32, 32, 3, device='cuda', dtype=torch.bfloat16)
        img_ids[..., 1] = torch.arange(32, device='cuda', dtype=torch.bfloat16).unsqueeze(1) / 31.0
        img_ids[..., 2] = torch.arange(32, device='cuda', dtype=torch.bfloat16).unsqueeze(0) / 31.0
        img_ids = img_ids.view(1, 1024, 3)

        # 批量提取v_pred
        v_pred_list = []
        for i in range(batch_size):
            if use_empty_prompt:
                pooled_proj = empty_prompt_cache['pooled_prompt_embeds']
                enc_states = empty_prompt_cache['prompt_embeds']
                txt_ids = empty_prompt_cache['text_ids']
            else:
                pooled_proj = pooled_embeds_list[i]
                enc_states = prompt_embeds_list[i]
                txt_ids = text_ids_list[i]

            v_pred = pipe.transformer(
                hidden_states=z_0_list[i],
                timestep=torch.tensor([pipe.scheduler.sigmas[-1]], device="cuda", dtype=torch.bfloat16) / 1000,
                pooled_projections=pooled_proj,
                encoder_hidden_states=enc_states,
                txt_ids=txt_ids,
                img_ids=img_ids,
                return_dict=False,
            )[0]
            v_pred_list.append(v_pred)

    # 批量计算签名
    signatures = []
    W_spatial = W.view(32, 32, 64)
    for v_pred in v_pred_list:
        v_pred_spatial = v_pred.view(32, 32, 64)
        S = np.zeros((grid_size, grid_size))

        for i in range(grid_size):
            for j in range(grid_size):
                W_patch = W_spatial[i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size, :].flatten().float()
                v_patch = v_pred_spatial[i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size, :].flatten().float()
                S[i, j] = torch.nn.functional.cosine_similarity(
                    W_patch.unsqueeze(0), v_patch.unsqueeze(0)
                ).item()
        signatures.append(S)

    return signatures

# ==========================================
# 5. 提取所有签名（高性能模式）
# ==========================================
print("\n" + "="*60)
print("开始提取签名（高性能模式）...")
print("="*60)

extraction_results = {
    'experiment_name': config['experiment_name'],
    'timestamp': datetime.now().isoformat(),
    'signatures': {}
}

batch_size = config.get('batch_size', 4)

# 5.1 提取原始水印图像的签名 S_orig
if not only_s_tamp:
    print("\n1. 提取原始水印图像签名 (S_orig)...")
    for batch_start in tqdm(range(0, len(images_info), batch_size), desc="原始图像批次"):
        batch_end = min(batch_start + batch_size, len(images_info))
        batch_infos = images_info[batch_start:batch_end]

        pil_images = []
        prompt_embeds_list = []
        pooled_embeds_list = []
        text_ids_list = []

        for img_info in batch_infos:
            img_id = img_info['id']
            prompt = img_info['prompt']
            S_orig_path = os.path.join(watermark_extra_dir, f"{img_id}_S_orig.npy")

            if os.path.exists(S_orig_path):
                continue

            img_path = os.path.join(watermarked_dir, img_info['image_file'])
            img = Image.open(img_path).convert('RGB')
            pil_images.append(img)

            encoded = encoded_prompts_cache[prompt]
            prompt_embeds_list.append(encoded['prompt_embeds'])
            pooled_embeds_list.append(encoded['pooled_prompt_embeds'])
            text_ids_list.append(encoded['text_ids'])

        if not pil_images:
            continue

        signatures = extract_signature_batch(pil_images, prompt_embeds_list, pooled_embeds_list, text_ids_list)

        for idx, img_info in enumerate(batch_infos):
            img_id = img_info['id']
            S_orig_path = os.path.join(watermark_extra_dir, f"{img_id}_S_orig.npy")
            if os.path.exists(S_orig_path):
                continue
            np.save(S_orig_path, signatures[idx])
else:
    print("\n1. 跳过 S_orig 提取（--only-s-tamp 模式）")

# 5.2 提取无水印图像的签名（负样本）
if not only_s_tamp:
    print("\n2. 提取无水印图像签名 (负样本)...")
    if os.path.exists(no_wm_summary_path):
        no_wm_images = no_wm_summary['images']

        for batch_start in tqdm(range(0, len(no_wm_images), batch_size), desc="无水印图像批次"):
            batch_end = min(batch_start + batch_size, len(no_wm_images))
            batch_infos = no_wm_images[batch_start:batch_end]

            pil_images = []
            prompt_embeds_list = []
            pooled_embeds_list = []
            text_ids_list = []

            for img_info in batch_infos:
                img_id = img_info['id']
                prompt = img_info['prompt']
                S_no_wm_path = os.path.join(watermark_extra_dir, f"{img_id}_S_no_wm.npy")

                if os.path.exists(S_no_wm_path):
                    continue

                img_path = os.path.join(no_watermark_dir, img_info['image_file'])
                img = Image.open(img_path).convert('RGB')
                pil_images.append(img)

                encoded = encoded_prompts_cache.get(prompt)
                if encoded is None:
                    encoded = encoded_prompts_cache.get(watermark_summary['images'][0]['prompt'])

                prompt_embeds_list.append(encoded['prompt_embeds'])
                pooled_embeds_list.append(encoded['pooled_prompt_embeds'])
                text_ids_list.append(encoded['text_ids'])

            if not pil_images:
                continue

            signatures = extract_signature_batch(pil_images, prompt_embeds_list, pooled_embeds_list, text_ids_list)

            for idx, img_info in enumerate(batch_infos):
                img_id = img_info['id']
                S_no_wm_path = os.path.join(watermark_extra_dir, f"{img_id}_S_no_wm.npy")
                if os.path.exists(S_no_wm_path):
                    continue
                np.save(S_no_wm_path, signatures[idx])
else:
    print("\n2. 跳过 S_no_wm 提取（--only-s-tamp 模式）")

# 5.3 提取真实自然图像的签名（负样本，增强学术严谨性）
if not only_s_tamp:
    real_image_dir = os.path.join(config['output_base_dir'], 'pic', 'real_images')
    if os.path.exists(real_image_dir) and os.listdir(real_image_dir):
        print("\n3. 提取真实自然图像签名 (负样本)...")
        real_images = [f for f in os.listdir(real_image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        # 使用第一个水印图像的 prompt 进行编码（真实图像没有对应 prompt）
        fallback_prompt = watermark_summary['images'][0]['prompt']
        fallback_encoded = encoded_prompts_cache.get(fallback_prompt)

        for batch_start in tqdm(range(0, len(real_images), batch_size), desc="真实图像批次"):
            batch_end = min(batch_start + batch_size, len(real_images))
            batch_files = real_images[batch_start:batch_end]

            pil_images = []
            prompt_embeds_list = []
            pooled_embeds_list = []
            text_ids_list = []

            for img_file in batch_files:
                img_id = f"real_{os.path.splitext(img_file)[0]}"
                S_real_path = os.path.join(watermark_extra_dir, f"{img_id}_S_real.npy")

                if os.path.exists(S_real_path):
                    continue

                img_path = os.path.join(real_image_dir, img_file)
                try:
                    img = Image.open(img_path).convert('RGB')
                    pil_images.append(img)
                    prompt_embeds_list.append(fallback_encoded['prompt_embeds'])
                    pooled_embeds_list.append(fallback_encoded['pooled_prompt_embeds'])
                    text_ids_list.append(fallback_encoded['text_ids'])
                except Exception as e:
                    print(f"\n   ❌ {img_file} 加载失败: {e}")
                    continue

            if not pil_images:
                continue

            signatures = extract_signature_batch(pil_images, prompt_embeds_list, pooled_embeds_list, text_ids_list)
            idx = 0
            for img_file in batch_files:
                img_id = f"real_{os.path.splitext(img_file)[0]}"
                S_real_path = os.path.join(watermark_extra_dir, f"{img_id}_S_real.npy")
                if os.path.exists(S_real_path):
                    continue
                np.save(S_real_path, signatures[idx])
                idx += 1
    else:
        print(f"\n3. 跳过真实图像提取（未找到目录: {real_image_dir}）")

# 5.4 提取攻击后图像的签名 S_tamp
print("\n4. 提取攻击后图像签名 (S_tamp)...")
for attack_name in tqdm(attack_configs, desc="攻击类型"):
    attack_results = attack_summary.get('images', [])

    # 如果 attack_summary 中没有详细的 images 信息，直接用原始图像信息列表
    if not attack_results:
        attack_results = images_info

    for batch_start in range(0, len(attack_results), batch_size):
        batch_end = min(batch_start + batch_size, len(attack_results))
        batch_infos = attack_results[batch_start:batch_end]

        pil_images = []
        prompt_embeds_list = []
        pooled_embeds_list = []
        text_ids_list = []
        img_ids_batch = []

        for img_info in batch_infos:
            img_id = img_info['id']
            prompt = img_info['prompt']

            # 兼容传统攻击（路径在 summary 中）和深度学习攻击（默认文件名）
            if attack_name in img_info.get('attacks', {}):
                attacked_img_name = img_info['attacks'][attack_name].get('image_file', f"{img_id}.png")
            else:
                attacked_img_name = f"{img_id}.png"

            attacked_img_path = os.path.join(attack_dir, attack_name, attacked_img_name)

            # 如果 png 不存在，试试 jpg
            if not os.path.exists(attacked_img_path):
                attacked_img_path = os.path.join(attack_dir, attack_name, f"{img_id}.jpg")

            if not os.path.exists(attacked_img_path):
                continue

            S_tamp_path = os.path.join(watermark_extra_dir, f"{img_id}_{attack_name}_S_tamp.npy")
            if os.path.exists(S_tamp_path) and os.path.getsize(S_tamp_path) > 0:
                continue

            try:
                img = Image.open(attacked_img_path).convert('RGB')

                # 对于裁剪攻击，做 edge padding 回原始尺寸，保持 VAE 中心 latent 不受污染
                if attack_name.startswith('crop_'):
                    w, h = img.size
                    orig_w = config.get('width', 512)
                    orig_h = config.get('height', 512)
                    pad_left = (orig_w - w) // 2
                    pad_top = (orig_h - h) // 2
                    pad_right = orig_w - w - pad_left
                    pad_bottom = orig_h - h - pad_top
                    padded_np = np.pad(np.array(img), ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), mode='edge')
                    img = Image.fromarray(padded_np)

                pil_images.append(img)

                encoded = encoded_prompts_cache[prompt]
                prompt_embeds_list.append(encoded['prompt_embeds'])
                pooled_embeds_list.append(encoded['pooled_prompt_embeds'])
                text_ids_list.append(encoded['text_ids'])
                img_ids_batch.append(img_id)
            except Exception as e:
                print(f"\n   ❌ {img_id} {attack_name} 加载失败: {e}")
                continue

        if not pil_images:
            continue

        try:
            signatures = extract_signature_batch(pil_images, prompt_embeds_list, pooled_embeds_list, text_ids_list)

            for idx, img_id in enumerate(img_ids_batch):
                S_tamp_path = os.path.join(watermark_extra_dir, f"{img_id}_{attack_name}_S_tamp.npy")
                np.save(S_tamp_path, signatures[idx])
        except Exception as e:
            print(f"\n   ❌ {attack_name} 批量提取失败: {e}")
            continue

# ==========================================
# 6. 保存提取摘要
# ==========================================
summary_path = os.path.join(watermark_extra_dir, 'extraction_summary.json')
existing_summary = {}
if os.path.exists(summary_path):
    with open(summary_path, 'r') as f:
        existing_summary = json.load(f)

existing_summary.update({
    'experiment_name': config['experiment_name'],
    'timestamp': datetime.now().isoformat(),
    'config': config,
    'num_images': len(images_info),
    'num_attacks': len(attack_configs),
    'attack_types': attack_configs,
    'output_dir': watermark_extra_dir
})

with open(summary_path, 'w') as f:
    json.dump(existing_summary, f, indent=2)

print(f"\n" + "="*60)
print("✅ 签名提取完成（高性能模式）！")
print("="*60)
print(f"   签名保存位置: {watermark_extra_dir}")
print(f"   提取摘要: {summary_path}")
print(f"\n   提取的签名文件:")
print(f"   - S_orig: {len([f for f in os.listdir(watermark_extra_dir) if f.endswith('_S_orig.npy')])} 个")
print(f"   - S_no_wm: {len([f for f in os.listdir(watermark_extra_dir) if f.endswith('_S_no_wm.npy')])} 个")
print(f"   - S_real: {len([f for f in os.listdir(watermark_extra_dir) if f.endswith('_S_real.npy')])} 个")
print(f"   - S_tamp: {len([f for f in os.listdir(watermark_extra_dir) if f.endswith('_S_tamp.npy')])} 个")
print(f"   最终显存使用: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
print(f"\n   🚀 高性能模式：保持所有模型和缓存常驻GPU！")
print(f"   现在可以运行: python3 analyze_robustness_result.py")
