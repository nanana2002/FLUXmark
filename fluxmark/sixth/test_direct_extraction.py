#!/usr/bin/env python3
"""
测试 Direct VAE-Latent Extraction 新逻辑。
使用原始的 key_W (spatial 形状 1x16x64x64) 和已生成的攻击图像。
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
import matplotlib.pyplot as plt

device = 'cuda:0'
model_path = config.get('model_path', '/home/daiyn/project_flux/model/flux-schnell')

# ==========================================
# 1. 加载 VAE（不需要 Transformer）
# ==========================================
print("🚀 仅加载 VAE (不需要加载庞大的 FLUX Transformer)...")
from diffusers import AutoencoderKL
vae = AutoencoderKL.from_pretrained(model_path, subfolder="vae", torch_dtype=torch.bfloat16).to(device)
torch.cuda.empty_cache()
print("✅ VAE 已加载")

# ==========================================
# 2. 加载原始 key_W (spatial 形状 1x16x64x64)
# ==========================================
key_W_path = "pic/baseline_gaussionshading_img/baseline_gaussionshading_key_W.npy"
key_W = torch.from_numpy(np.load(key_W_path)).to(device, dtype=torch.bfloat16)  # (1, 16, 64, 64)
print(f"✅ 已加载 key_W, shape: {key_W.shape}")

# 将 key_W 转换为 packed 形式 (1, 1024, 64)，再 reshape 为 (32, 32, 64)
def pack_latents(latents):
    b, c, h, w = latents.shape
    latents = latents.view(b, c, h // 2, 2, w // 2, 2)
    latents = latents.permute(0, 2, 4, 1, 3, 5)
    return latents.reshape(b, (h // 2) * (w // 2), c * 4)

key_W_packed = pack_latents(key_W)  # (1, 1024, 64)
W_spatial = key_W_packed.view(32, 32, 64)
print(f"✅ key_W packed shape: {key_W_packed.shape}")

# ==========================================
# 3. Direct VAE 提取函数
# ==========================================
def direct_latent_extraction(image_path):
    img = Image.open(image_path).convert("RGB")
    # 统一 resize 到 512x512，确保和 key_W 的 latent 尺寸匹配
    if img.size != (512, 512):
        img = img.resize((512, 512), Image.LANCZOS)
    img_tensor = T.ToTensor()(img).unsqueeze(0).to(device, dtype=torch.bfloat16)
    img_tensor = (img_tensor - 0.5) * 2.0
    
    with torch.no_grad():
        z_0 = vae.encode(img_tensor).latent_dist.sample()
        z_0 = (z_0 - vae.config.shift_factor) * vae.config.scaling_factor
        z_0_packed = pack_latents(z_0)  # (1, 1024, 64)
    
    z_0_spatial = z_0_packed.view(32, 32, 64)
    
    # 在 32x32 网格上逐点计算余弦相似度
    S_direct = np.zeros((32, 32))
    for i in range(32):
        for j in range(32):
            w_token = W_spatial[i, j, :].flatten().float()
            z_token = z_0_spatial[i, j, :].flatten().float()
            S_direct[i, j] = torch.nn.functional.cosine_similarity(
                w_token.unsqueeze(0), z_token.unsqueeze(0)
            ).item()
    
    return S_direct

# ==========================================
# 4. 测试多张图像
# ==========================================
base_dir = "pic/baseline_gaussionshading_img"
attacks_dir = os.path.join(base_dir, "attacks")

test_cases = [
    ("原始图像", os.path.join(base_dir, "baseline_gaussionshading_flux.png")),
    ("黑块中心", os.path.join(attacks_dir, "black_block_center.png")),
    ("黑块随机", os.path.join(attacks_dir, "black_block_random.png")),
    ("JPEG 50", os.path.join(attacks_dir, "jpeg_50.jpg")),
    ("SDEdit 0.3", os.path.join(attacks_dir, "sdedit_0.3.png")),
    ("SDXL 油画", os.path.join(attacks_dir, "sdxl_style_oil_painting", "baseline.png")),
    ("FluxFill 中心", os.path.join(attacks_dir, "fluxfill_center", "baseline.png")),
]

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.flatten()

for idx, (label, path) in enumerate(test_cases):
    if not os.path.exists(path):
        print(f"⚠️  跳过 (未找到): {label} -> {path}")
        continue
    
    print(f"\n🔬 测试: {label}")
    S = direct_latent_extraction(path)
    mean_score = np.mean(S)
    std_score = np.std(S)
    print(f"   全局均值: {mean_score:.4f} | 标准差: {std_score:.4f} | 最小值: {np.min(S):.4f} | 最大值: {np.max(S):.4f}")
    
    ax = axes[idx]
    im = ax.imshow(S, cmap='viridis', interpolation='nearest', vmin=-0.2, vmax=0.5)
    ax.set_title(f"{label}\nmean={mean_score:.3f}", fontsize=10)
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

# 隐藏多余的子图
for idx in range(len(test_cases), len(axes)):
    axes[idx].axis('off')

plt.suptitle("Direct VAE-Latent Extraction (32x32 grid)", fontsize=14)
plt.tight_layout()
out_path = "direct_extraction_heatmap.png"
plt.savefig(out_path, dpi=200, bbox_inches='tight')
print(f"\n✅ 热力图已保存: {out_path}")
