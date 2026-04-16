#!/usr/bin/env python3
"""
网格分辨率对比可视化脚本（修复OOM终极版）
同一张图只提取一次32×32签名，16/8通过下采样得到
大幅降低显存占用，A100-80GB稳跑
"""

import os
import json

with open("config.json", "r") as f:
    config = json.load(f)
os.environ["CUDA_VISIBLE_DEVICES"] = "3"  # 强制跑在GPU3
os.environ["HF_HOME"] = config["hf_cache"]

import torch
from diffusers import FluxPipeline
import torchvision.transforms as T
from PIL import Image
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cv2

# ========== 用户可配置 ==========
IMG_IDX = 27
ATTACK_NAME = "black_block_random"
OUTPUT_PATH = os.path.join(
    config["output_base_dir"], "result", "ablation_grid", "grid_size_comparison.png"
)
# ================================

watermarked_dir = os.path.join(config["output_base_dir"], "pic", "watermarked_img")
attack_dir = os.path.join(config["output_base_dir"], "pic", "attack_watermarked_img")

summary_path = os.path.join(
    watermarked_dir, f"{config['experiment_name']}_watermark_summary.json"
)
with open(summary_path, "r") as f:
    summary = json.load(f)
img_info = summary["images"][IMG_IDX]
img_id = img_info["id"]
prompt = img_info["prompt"]

orig_img_path = os.path.join(watermarked_dir, img_info["image_file"])
attacked_img_path = os.path.join(attack_dir, ATTACK_NAME, f"{img_id}.png")
if not os.path.exists(attacked_img_path):
    attacked_img_path = os.path.join(attack_dir, ATTACK_NAME, f"{img_id}.jpg")

print(f"📷 原始图: {orig_img_path}")
print(f"🔥 攻击图: {attacked_img_path}")

orig_img = Image.open(orig_img_path).convert("RGB").resize((512, 512), Image.LANCZOS)
attacked_img = Image.open(attacked_img_path).convert("RGB").resize((512, 512), Image.LANCZOS)

# ========== 1. 加载模型并编码 prompt ==========
print("\n🚀 加载 FLUX 模型...")
pipe = FluxPipeline.from_pretrained(config["model_path"], torch_dtype=torch.bfloat16)
pipe.enable_attention_slicing(slice_size="auto")
pipe.to("cpu")

pipe.text_encoder.to("cuda")
pipe.text_encoder_2.to("cuda")
torch.cuda.empty_cache()

with torch.no_grad():
    text_inputs = pipe.tokenizer(
        prompt, padding="max_length", max_length=pipe.tokenizer.model_max_length,
        truncation=True, return_tensors="pt",
    )
    clip_embeds = pipe.text_encoder(text_inputs.input_ids.to("cuda"), output_hidden_states=False)[0]
    pooled_prompt_embeds = clip_embeds[0:1, -1]

    text_inputs_2 = pipe.tokenizer_2(
        prompt, padding="max_length", max_length=256,
        truncation=True, return_tensors="pt",
    )
    prompt_embeds = pipe.text_encoder_2(text_inputs_2.input_ids.to("cuda"), output_hidden_states=False)[0]
    text_ids = torch.zeros(prompt_embeds.shape[1], 3, device="cuda", dtype=prompt_embeds.dtype)

encoded = {
    "prompt_embeds": prompt_embeds.cpu(),
    "pooled_prompt_embeds": pooled_prompt_embeds.cpu(),
    "text_ids": text_ids.cpu(),
}

# 释放文本编码器，彻底省显存
del pipe.text_encoder, pipe.text_encoder_2
del prompt_embeds, pooled_prompt_embeds, text_ids, clip_embeds
torch.cuda.empty_cache()

pipe.vae.to("cuda")
pipe.transformer.to("cuda")
print("✅ VAE + Transformer 已加载到 GPU")

# ========== 2. 重建密码本 W ==========
secret_key = config["secret_key"]
g = torch.Generator(device="cuda").manual_seed(secret_key)
W_raw = torch.randn((1, 1024, 64), generator=g, device="cuda", dtype=torch.float32)
W_spatial = W_raw.view(1, 32, 32, 64)
F_W = torch.fft.fftshift(torch.fft.fft2(W_spatial, dim=(1, 2)), dim=(1, 2))
h, w = 32, 32
Y, X = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
center_y, center_x = h // 2, w // 2
radius = torch.sqrt((Y - center_y) ** 2 + (X - center_x) ** 2).to("cuda")
r_inner = config["fft_radius_inner"]
r_outer = config["fft_radius_outer"]
mask_fft = ((radius >= r_inner) & (radius <= r_outer)).float().unsqueeze(0).unsqueeze(-1)
F_W_filtered = F_W * mask_fft
W_filtered = torch.fft.ifft2(torch.fft.ifftshift(F_W_filtered, dim=(1, 2)), dim=(1, 2)).real
W = W_filtered.view(1, 1024, 64).to(torch.bfloat16)
W = W / (torch.norm(W, dim=-1, keepdim=True) + 1e-8)
W_s = W.view(32, 32, 64)

# ========== 3. 签名提取函数（只提取32×32，最省显存） ==========
def extract_32x32_signature(img_pil):
    img_tensor = T.ToTensor()(img_pil).unsqueeze(0).to("cuda", dtype=torch.bfloat16)
    img_tensor = (img_tensor - 0.5) * 2.0

    pe = encoded["prompt_embeds"].to("cuda")
    ppe = encoded["pooled_prompt_embeds"].to("cuda")
    tid = encoded["text_ids"].to("cuda")

    with torch.no_grad():
        z_0 = pipe.vae.encode(img_tensor).latent_dist.sample()
        z_0 = (z_0 - pipe.vae.config.shift_factor) * pipe.vae.config.scaling_factor
        z_0 = pipe._pack_latents(z_0, batch_size=1, num_channels_latents=16, height=64, width=64)

        img_ids = torch.zeros(32, 32, 3, device="cuda", dtype=torch.bfloat16)
        img_ids[..., 1] = torch.arange(32, device="cuda", dtype=torch.bfloat16).unsqueeze(1) / 31.0
        img_ids[..., 2] = torch.arange(32, device="cuda", dtype=torch.bfloat16).unsqueeze(0) / 31.0
        img_ids = img_ids.view(1, 1024, 3)

        v_pred = pipe.transformer(
            hidden_states=z_0,
            timestep=torch.tensor([pipe.scheduler.sigmas[-1]], device="cuda", dtype=torch.bfloat16) / 1000,
            pooled_projections=ppe,
            encoder_hidden_states=pe,
            txt_ids=tid,
            img_ids=img_ids,
            return_dict=False,
        )[0]

    v_pred_spatial = v_pred.view(32, 32, 64)
    S = np.zeros((32, 32))

    for i in range(32):
        for j in range(32):
            wp = W_s[i, j].flatten().float()
            vp = v_pred_spatial[i, j].flatten().float()
            S[i, j] = torch.nn.functional.cosine_similarity(wp.unsqueeze(0), vp.unsqueeze(0)).item()

    # 强制释放所有临时张量
    del img_tensor, z_0, v_pred, v_pred_spatial, pe, ppe, tid, img_ids
    torch.cuda.empty_cache()
    return S

# ========== 4. 只提取一次 32×32，16/8 下采样得到（核心OOM修复） ==========
print("\n🔬 提取 32×32 签名（原图+攻击图）...")
S_orig_32 = extract_32x32_signature(orig_img)
S_tamp_32 = extract_32x32_signature(attacked_img)

def downsample_sig(s, size):
    return cv2.resize(s, (size, size), interpolation=cv2.INTER_NEAREST)

results = {}
for grid_size in [32, 16, 8]:
    if grid_size == 32:
        s_orig = S_orig_32
        s_tamp = S_tamp_32
    else:
        s_orig = downsample_sig(S_orig_32, grid_size)
        s_tamp = downsample_sig(S_tamp_32, grid_size)
    
    diff = np.abs(s_orig - s_tamp)
    results[grid_size] = {
        "diff": diff,
        "std": float(np.std(diff))
    }
    print(f"   {grid_size}×{grid_size} | std = {results[grid_size]['std']:.4f}")

# ========== 5. 可视化 ==========
fig, axes = plt.subplots(2, 4, figsize=(20, 10))

axes[0, 0].imshow(orig_img)
axes[0, 0].set_title("Original Watermarked", fontsize=14)
axes[0, 0].axis("off")

axes[0, 1].imshow(attacked_img)
axes[0, 1].set_title(f"Attacked: {ATTACK_NAME}", fontsize=14)
axes[0, 1].axis("off")

true_mask_path = os.path.join(attack_dir, ATTACK_NAME, f"{img_id}_mask.npy")
true_mask_8x8 = np.load(true_mask_path) if os.path.exists(true_mask_path) else None

if true_mask_8x8 is not None:
    axes[0, 2].imshow(true_mask_8x8, cmap="Reds", interpolation="nearest")
    axes[0, 2].set_title("True Mask (8×8)", fontsize=14)
else:
    axes[0, 2].text(0.5, 0.5, "No Mask", ha="center", va="center", fontsize=14)
axes[0, 2].axis("off")

axes[0, 3].axis("off")
axes[0, 3].text(
    0.5, 0.5,
    f"Grid Comparison\n(img: {img_id})\n"
    f"32×32 std={results[32]['std']:.3f}\n"
    f"16×16 std={results[16]['std']:.3f}\n"
    f"8×8  std={results[8]['std']:.3f}",
    ha="center", va="center", fontsize=14, transform=axes[0, 3].transAxes,
    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5)
)

for idx, grid_size in enumerate([32, 16, 8]):
    ax = axes[1, idx]
    im = ax.imshow(results[grid_size]["diff"], cmap="hot", interpolation="nearest")
    ax.set_title(f"{grid_size}×{grid_size} | std={results[grid_size]['std']:.3f}", fontsize=14)
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

ax_large = axes[1, 3]
heatmap_512 = np.kron(results[8]["diff"], np.ones((64, 64)))
im_large = ax_large.imshow(heatmap_512, cmap="hot")
ax_large.set_title("8×8 Upsampled (512×512)", fontsize=14)
ax_large.axis("off")
plt.colorbar(im_large, ax=ax_large, fraction=0.046, pad=0.04)

plt.tight_layout()
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight")
print(f"\n✅ 对比图已保存: {OUTPUT_PATH}")

# 最终释放
del pipe, W, W_s, encoded
torch.cuda.empty_cache()