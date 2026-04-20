#!/usr/bin/env python3
"""
水印图像生成脚本（极致性能版 - 50GB显存全速运行）
优化策略：
- 保持所有模型常驻GPU
- 批量推理（batch inference）
- 预编码prompts缓存于GPU
- FFT密码本预计算并常驻GPU
- 不调用显存清理，最大化利用显存换速度
"""

import os
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

output_dir = os.path.join(config['output_base_dir'], 'pic', 'watermarked_img')
os.makedirs(output_dir, exist_ok=True)

print(f"🎮 使用 GPU: {config.get('gpu_id', 0)}")

# 加载prompts
prompts_path = os.path.join(config['output_base_dir'], 'prompts.json')
with open(prompts_path, 'r') as f:
    prompts_data = json.load(f)
prompts = prompts_data['prompts']

# 根据 config 中的 num_samples 截断（方便小批量测试）
num_samples = config.get('num_samples', len(prompts))
prompts = prompts[:num_samples]
print(f"📋 加载了 {len(prompts)} 条 prompts（config.num_samples={num_samples}）")

# ==========================================
# 1. 加载模型（直接到GPU，保持常驻）
# ==========================================
print("\n🚀 加载 FLUX 模型到GPU...")
pipe = FluxPipeline.from_pretrained(
    config['model_path'],
    torch_dtype=torch.bfloat16
)

# 性能优化
pipe.enable_attention_slicing(slice_size="auto")

# 直接加载到GPU并保持常驻
pipe.to("cuda")
print(f"   模型已加载到GPU，显存使用: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

# ==========================================
# 2. 预编码所有prompts（缓存于GPU）
# ==========================================
print("🔤 预编码所有prompts到GPU缓存...")
encoded_prompts = []
for prompt in tqdm(prompts, desc="编码进度"):
    with torch.no_grad():
        prompt_embeds, pooled_prompt_embeds, text_ids = pipe.encode_prompt(
            prompt=prompt, prompt_2=None, max_sequence_length=256
        )
    # 保持在GPU上缓存
    encoded_prompts.append({
        'prompt_embeds': prompt_embeds,
        'pooled_prompt_embeds': pooled_prompt_embeds,
        'text_ids': text_ids
    })

print(f"   已编码 {len(encoded_prompts)} 个prompts到GPU")
print(f"   当前显存使用: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

# ==========================================
# 3. 生成FFT密码本W（预计算并常驻GPU）
# ==========================================
print("🔐 生成FFT密码本...")

secret_key = config['secret_key']
generator = torch.Generator(device='cuda').manual_seed(secret_key)
W_raw = torch.randn((1, 1024, 64), generator=generator, device='cuda', dtype=torch.float32)
W_spatial = W_raw.view(1, 32, 32, 64)

# FFT变换
F_W = torch.fft.fftshift(torch.fft.fft2(W_spatial, dim=(1, 2)), dim=(1, 2))
h, w = 32, 32
Y, X = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
center_y, center_x = h // 2, w // 2
radius = torch.sqrt((Y - center_y)**2 + (X - center_x)**2).to('cuda')

# 带通滤波器
r_inner = config['fft_radius_inner']
r_outer = config['fft_radius_outer']
mask_fft = ((radius >= r_inner) & (radius <= r_outer)).float().unsqueeze(0).unsqueeze(-1)

F_W_filtered = F_W * mask_fft
W_filtered = torch.fft.ifft2(torch.fft.ifftshift(F_W_filtered, dim=(1, 2)), dim=(1, 2)).real
W = W_filtered.view(1, 1024, 64).to(torch.bfloat16)

# L2归一化
W = W / (torch.norm(W, dim=-1, keepdim=True) + 1e-8)

print(f"   FFT带通: [{r_inner}, {r_outer}]")
print(f"   密码本已常驻GPU（全图统一注入），显存使用: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

# ==========================================
# 4. 正交注入回调函数
# ==========================================
alpha = config['alpha']

def orthogonal_drift_callback(pipe, step_index, timestep, callback_kwargs):
    """正交注入回调（全图统一注入）"""
    latents = callback_kwargs["latents"]

    # Token-wise正交化
    dot_W_x = torch.sum(W * latents, dim=-1, keepdim=True)
    dot_x_x = torch.sum(latents * latents, dim=-1, keepdim=True) + 1e-8
    W_perp = W - (dot_W_x / dot_x_x) * latents

    # 能量重归一化
    norm_W = torch.norm(W, dim=-1, keepdim=True)
    norm_W_perp = torch.norm(W_perp, dim=-1, keepdim=True) + 1e-8
    W_perp = W_perp * (norm_W / norm_W_perp)

    # 欧拉积分加权
    sigmas = pipe.scheduler.sigmas
    dt = sigmas[step_index + 1] - sigmas[step_index]
    latents = latents + alpha * W_perp * abs(dt)

    callback_kwargs["latents"] = latents
    return callback_kwargs

# ==========================================
# 5. 批量生成水印图像（高性能模式）
# ==========================================
print(f"\n🎨 开始生成 {len(prompts)} 张水印图像（高性能模式）...\n")

results_summary = {
    'experiment_name': config['experiment_name'],
    'timestamp': datetime.now().isoformat(),
    'total_images': len(prompts),
    'config': config,
    'images': []
}

batch_size = config.get('batch_size', 4)

# 批量生成
for batch_start in tqdm(range(0, len(prompts), batch_size), desc="生成批次"):
    batch_end = min(batch_start + batch_size, len(prompts))
    batch_indices = list(range(batch_start, batch_end))

    # 收集batch数据
    batch_prompt_embeds = torch.cat([encoded_prompts[i]['prompt_embeds'] for i in batch_indices])
    batch_pooled_embeds = torch.cat([encoded_prompts[i]['pooled_prompt_embeds'] for i in batch_indices])
    batch_text_ids = torch.cat([encoded_prompts[i]['text_ids'] for i in batch_indices])

    # 批量生成带水印图像
    with torch.no_grad():
        images = pipe(
            prompt_embeds=batch_prompt_embeds,
            pooled_prompt_embeds=batch_pooled_embeds,
            num_inference_steps=config['num_inference_steps'],
            guidance_scale=config['guidance_scale'],
            height=config['height'],
            width=config['width'],
            callback_on_step_end=orthogonal_drift_callback,
            num_images_per_prompt=1,
        ).images

    # 批量提取签名
    for idx, img_idx in enumerate(batch_indices):
        img_id = f"wm_{img_idx:03d}"
        img_filename = f"{img_id}.png"
        img_path = os.path.join(output_dir, img_filename)
        images[idx].save(img_path)

        # ==========================================
        # 6. 提取签名 S_orig
        # ==========================================
        img_tensor = T.ToTensor()(images[idx]).unsqueeze(0).to("cuda", dtype=torch.bfloat16)
        img_tensor = (img_tensor - 0.5) * 2.0

        # 使用原始 encoded_prompts 避免维度问题
        encoded = encoded_prompts[img_idx]
        prompt_embeds = encoded['prompt_embeds']
        pooled_prompt_embeds = encoded['pooled_prompt_embeds']
        text_ids = encoded['text_ids']

        with torch.no_grad():
            # VAE编码
            z_0 = pipe.vae.encode(img_tensor).latent_dist.sample()
            z_0 = (z_0 - pipe.vae.config.shift_factor) * pipe.vae.config.scaling_factor
            z_0 = pipe._pack_latents(z_0, batch_size=1, num_channels_latents=16, height=64, width=64)

            # 构造img_ids
            h = w = 32
            img_ids = torch.zeros(h, w, 3, device='cuda', dtype=torch.bfloat16)
            img_ids[..., 1] = torch.arange(h, device='cuda', dtype=torch.bfloat16).unsqueeze(1) / max(h - 1, 1)
            img_ids[..., 2] = torch.arange(w, device='cuda', dtype=torch.bfloat16).unsqueeze(0) / max(w - 1, 1)
            img_ids = img_ids.view(1, h * w, 3)

            # Transformer提取v_pred
            t_detect = torch.tensor([pipe.scheduler.sigmas[-1]], device="cuda", dtype=torch.bfloat16)
            v_pred = pipe.transformer(
                hidden_states=z_0, timestep=t_detect / 1000,
                pooled_projections=pooled_prompt_embeds,
                encoder_hidden_states=prompt_embeds,
                txt_ids=text_ids,
                img_ids=img_ids,
                return_dict=False,
            )[0]

        # 计算32x32签名（潜空间原生分辨率，每个token独立计算）
        v_pred_spatial = v_pred.view(32, 32, 64)
        W_spatial = W.view(32, 32, 64)
        S_orig = np.zeros((32, 32))

        for i in range(32):
            for j in range(32):
                W_patch = W_spatial[i, j, :].flatten().float()
                v_patch = v_pred_spatial[i, j, :].flatten().float()
                S_orig[i, j] = torch.nn.functional.cosine_similarity(
                    W_patch.unsqueeze(0), v_patch.unsqueeze(0)
                ).item()

        # 保存签名
        sig_filename = f"{img_id}_signature.npy"
        sig_path = os.path.join(output_dir, sig_filename)
        np.save(sig_path, S_orig)

        # 全图统计
        img_result = {
            'id': img_id,
            'prompt': prompts[img_idx],
            'image_file': img_filename,
            'signature_file': sig_filename,
            'stats': {
                'mean': float(np.mean(S_orig)),
                'std': float(np.std(S_orig)),
                'min': float(np.min(S_orig)),
                'max': float(np.max(S_orig))
            }
        }
        results_summary['images'].append(img_result)

# ==========================================
# 7. 保存实验摘要
# ==========================================
summary_path = os.path.join(output_dir, f'{config["experiment_name"]}_watermark_summary.json')
with open(summary_path, 'w') as f:
    json.dump(results_summary, f, indent=2)

print(f"\n✅ 水印图像生成完成！")
print(f"   图像保存位置: {output_dir}")
print(f"   实验摘要: {summary_path}")
print(f"\n📊 全图签名统计:")
means = [img['stats']['mean'] for img in results_summary['images']]
print(f"   平均值: {np.mean(means):.4f}")
print(f"   标准差: {np.std(means):.4f}")
print(f"   最小值: {np.min(means):.4f}")
print(f"   最大值: {np.max(means):.4f}")
print(f"\n   最终显存使用: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
print(f"   🚀 高性能模式：保持所有模型和缓存常驻GPU！")
