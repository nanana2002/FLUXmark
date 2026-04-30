#!/usr/bin/env python3
"""
验证：用相同的 prompt 重新提取原始水印图像的签名，能否替代存储的 S_orig？
如果 re-extracted S 和保存的 S_orig 高度一致，就不需要数据库了！
"""

import os
import json

with open('config.json', 'r') as f:
    config = json.load(f)

os.environ['CUDA_VISIBLE_DEVICES'] = str(config.get('gpu_id', 0))
os.environ['HF_HOME'] = config.get('hf_cache', '/home/daiyn/hf_cache')

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from diffusers import FluxPipeline

base_dir = "watermark_extra"
watermarked_dir = "pic/watermarked_img"

# 加载 S_orig
img_id = "wm_000"
S_orig_saved = np.load(os.path.join(base_dir, f"{img_id}_S_orig.npy"))
print(f"S_orig saved shape: {S_orig_saved.shape}, mean={S_orig_saved.mean():.4f}, std={S_orig_saved.std():.4f}")

# 加载实验摘要获取 prompt
summary_path = os.path.join(watermarked_dir, f'{config["experiment_name"]}_watermark_summary.json')
with open(summary_path, 'r') as f:
    summary = json.load(f)

prompt = summary['images'][0]['prompt']
print(f"Prompt: {prompt}")

# 加载模型（轻量模式：只加载一次）
print("\n🚀 加载 FLUX 模型...")
pipe = FluxPipeline.from_pretrained(config['model_path'], torch_dtype=torch.bfloat16)
pipe.to("cuda")
print("✅ 模型已加载")

# 编码 prompt
with torch.no_grad():
    prompt_embeds, pooled_prompt_embeds, text_ids = pipe.encode_prompt(
        prompt=prompt, prompt_2=None, max_sequence_length=256
    )

# 重建 W
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
print("✅ W 已重建")

# 重新提取原始图像的签名
img_path = os.path.join(watermarked_dir, f"{img_id}.png")
img = Image.open(img_path).convert("RGB")
img = img.resize((512, 512), Image.LANCZOS)
img_tensor = T.ToTensor()(img).unsqueeze(0).to("cuda", dtype=torch.bfloat16)
img_tensor = (img_tensor - 0.5) * 2.0

with torch.no_grad():
    z_0 = pipe.vae.encode(img_tensor).latent_dist.sample()
    z_0 = (z_0 - pipe.vae.config.shift_factor) * pipe.vae.config.scaling_factor
    z_0 = pipe._pack_latents(z_0, batch_size=1, num_channels_latents=16, height=64, width=64)
    
    h = w = 32
    img_ids = torch.zeros(h, w, 3, device='cuda', dtype=torch.bfloat16)
    img_ids[..., 1] = torch.arange(h, device='cuda', dtype=torch.bfloat16).unsqueeze(1) / max(h - 1, 1)
    img_ids[..., 2] = torch.arange(w, device='cuda', dtype=torch.bfloat16).unsqueeze(0) / max(w - 1, 1)
    img_ids = img_ids.view(1, h * w, 3)
    
    v_pred = pipe.transformer(
        hidden_states=z_0, timestep=torch.tensor([pipe.scheduler.sigmas[-1]], device="cuda", dtype=torch.bfloat16) / 1000,
        pooled_projections=pooled_prompt_embeds, encoder_hidden_states=prompt_embeds,
        txt_ids=text_ids, img_ids=img_ids, return_dict=False,
    )[0]

# 计算签名
v_pred_spatial = v_pred.view(32, 32, 64)
W_spatial = W.view(32, 32, 64)
S_reextract = np.zeros((32, 32))
for i in range(32):
    for j in range(32):
        W_patch = W_spatial[i, j, :].flatten().float()
        v_patch = v_pred_spatial[i, j, :].flatten().float()
        S_reextract[i, j] = torch.nn.functional.cosine_similarity(
            W_patch.unsqueeze(0), v_patch.unsqueeze(0)
        ).item()

print(f"\nS_reextract mean={S_reextract.mean():.4f}, std={S_reextract.std():.4f}")

# 对比
print(f"\n{'='*60}")
print("对比：保存的 S_orig vs 重新提取的 S_reextract")
print(f"{'='*60}")
print(f"Mean diff:     {np.abs(S_orig_saved.mean() - S_reextract.mean()):.6f}")
print(f"Std diff:      {np.abs(S_orig_saved.std() - S_reextract.std()):.6f}")
print(f"Max abs diff:  {np.abs(S_orig_saved - S_reextract).max():.6f}")
print(f"Mean abs diff: {np.abs(S_orig_saved - S_reextract).mean():.6f}")

# 计算两者的余弦相似度（作为整体签名的相似度）
flat_orig = torch.from_numpy(S_orig_saved.flatten()).float()
flat_re = torch.from_numpy(S_reextract.flatten()).float()
overall_sim = torch.nn.functional.cosine_similarity(flat_orig.unsqueeze(0), flat_re.unsqueeze(0)).item()
print(f"Overall cosine similarity: {overall_sim:.6f}")

# 如果足够接近，测试用 S_reextract 代替 S_orig 做差分定位的效果
print(f"\n{'='*60}")
print("用 S_reextract 代替 S_orig 做差分定位的效果")
print(f"{'='*60}")

for attack_name in ['black_block_center', 'black_block_random', 'copy_move', 'fluxfill_center']:
    S_tamp_path = os.path.join(base_dir, f"{img_id}_{attack_name}_S_tamp.npy")
    if not os.path.exists(S_tamp_path):
        continue
    S_tamp = np.load(S_tamp_path)
    
    # Oracle: S_orig - S_tamp
    S_diff_oracle = S_tamp - S_orig_saved
    # Re-extract: S_reextract - S_tamp
    S_diff_re = S_tamp - S_reextract
    
    # 参考 mask (Oracle)
    threshold = -2 * S_diff_oracle.std()
    mask_ref = S_diff_oracle < threshold
    
    # Re-extract mask
    threshold_re = -2 * S_diff_re.std()
    mask_re = S_diff_re < threshold_re
    
    overlap = np.logical_and(mask_ref, mask_re).sum()
    union = np.logical_or(mask_ref, mask_re).sum()
    iou = overlap / union if union > 0 else 0
    
    print(f"{attack_name:25s}  IoU={iou:.3f}  mask_coverage_ref={mask_ref.mean():.3f}  mask_coverage_re={mask_re.mean():.3f}")
