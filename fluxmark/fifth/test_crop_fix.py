#!/usr/bin/env python3
"""
快速验证 crop 攻击的 edge padding 修复是否有效
只提取 wm_000 的一张原图 + 一张 crop_0.75 图，打印签名均值
"""

import os
import sys
import json

with open('config.json', 'r') as f:
    config = json.load(f)
os.environ['CUDA_VISIBLE_DEVICES'] = str(config.get('gpu_id', 0))
os.environ['HF_HOME'] = config['hf_cache']

import torch
from diffusers import FluxPipeline
import torchvision.transforms as T
from PIL import Image
import numpy as np

watermarked_dir = os.path.join(config['output_base_dir'], 'pic', 'watermarked_img')
attack_dir = os.path.join(config['output_base_dir'], 'pic', 'attack_watermarked_img')

print("🚀 加载 FLUX 模型...")
pipe = FluxPipeline.from_pretrained(config['model_path'], torch_dtype=torch.bfloat16)
pipe.to("cuda")
# pipe.enable_vae_tiling()
pipe.enable_attention_slicing(slice_size="auto")

# 预编码 prompt
prompt = "a cat holding a sign that says hello world"  # 会被 wm_000 的真实 prompt 覆盖

# 读取 wm_000 的 prompt
summary_path = os.path.join(watermarked_dir, f'{config["experiment_name"]}_watermark_summary.json')
with open(summary_path, 'r') as f:
    summary = json.load(f)
prompt = summary['images'][0]['prompt']

with torch.no_grad():
    prompt_embeds, pooled_prompt_embeds, text_ids = pipe.encode_prompt(prompt=prompt, prompt_2=None, max_sequence_length=256)

# 重建密码本 W
secret_key = config['secret_key']
g = torch.Generator(device='cuda').manual_seed(secret_key)
W_raw = torch.randn((1, 1024, 64), generator=g, device='cuda', dtype=torch.float32)
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

def extract_single(img):
    """提取单张图的 8x8 签名"""
    img_tensor = T.ToTensor()(img).unsqueeze(0).to("cuda", dtype=torch.bfloat16)
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
            S[i, j] = torch.nn.functional.cosine_similarity(W_patch.unsqueeze(0), v_patch.unsqueeze(0)).item()
    return S

# 1. 原图
orig_img = Image.open(os.path.join(watermarked_dir, 'wm_000.png')).convert('RGB')
S_orig = extract_single(orig_img)

# 2. crop_0.75 原图（384x384）
crop_img = Image.open(os.path.join(attack_dir, 'crop_0.75', 'wm_000.png')).convert('RGB')

# 3. edge pad 回 512
w, h = crop_img.size
orig_w = config.get('width', 512)
orig_h = config.get('height', 512)
pad_left = (orig_w - w) // 2
pad_top = (orig_h - h) // 2
pad_right = orig_w - w - pad_left
pad_bottom = orig_h - h - pad_top
padded_edge = Image.fromarray(
    np.pad(np.array(crop_img), ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), mode='edge')
)
S_edge = extract_single(padded_edge)

# 4. gray pad 回 512（旧方案）
padded_gray = Image.new('RGB', (orig_w, orig_h), (128, 128, 128))
padded_gray.paste(crop_img, (pad_left, pad_top))
S_gray = extract_single(padded_gray)

# 5. resize 到 512（更旧的方案）
S_resize = extract_single(crop_img.resize((512, 512), Image.LANCZOS))

mask_8x8 = np.zeros((8, 8))
mask_8x8[1:7, 1:7] = 1.0

print("\n" + "="*50)
print("wm_000 签名均值对比（核心区域 mask_8x8）")
print("="*50)
print(f"原始完整图:        {np.mean(S_orig[mask_8x8==1.0]):.4f}")
print(f"Edge Padding:      {np.mean(S_edge[mask_8x8==1.0]):.4f}")
print(f"Gray Padding:      {np.mean(S_gray[mask_8x8==1.0]):.4f}")
print(f"Resize 到 512:     {np.mean(S_resize[mask_8x8==1.0]):.4f}")
print("="*50)
print("\n结论:")
print("  • Edge Padding 应该接近原始值 (~0.04)")
print("  • Gray/Resize 应该接近 0 甚至负数")
