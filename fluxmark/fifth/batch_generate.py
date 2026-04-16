#!/usr/bin/env python3
"""
OrthoFlow 规模化批量水印生成脚本
生成 20 张带水印的图像，保存签名和配置
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

os.environ['HF_HOME'] = '/data/daiyina/hf_cache'
output_dir = '/data/daiyina/project_flux/pic'
os.makedirs(output_dir, exist_ok=True)

# ==========================================
# 1. 加载实验配置
# ==========================================
print("📋 加载实验配置...")
with open('prompts.json', 'r') as f:
    config = json.load(f)

experiment_name = config['experiment_name']
prompts = config['prompts']
num_images = len(prompts)

print(f"实验名称: {experiment_name}")
print(f"生成数量: {num_images} 张")
print(f"FFT 带通: {config['fft_radius_inner']}-{config['fft_radius_outer']}")
print(f"注入强度 alpha: {config['alpha']}")

# ==========================================
# 2. 加载模型 (只加载一次)
# ==========================================
print("\n🚀 加载 FLUX 模型...")
pipe = FluxPipeline.from_pretrained(
    '/data/daiyina/project_flux/model/flux-schnell',
    torch_dtype=torch.bfloat16
)

# 预编码所有prompts，然后卸载encoder节省显存
print("🔤 预编码所有prompts...")
encoded_prompts = []
for prompt in prompts:
    with torch.no_grad():
        prompt_embeds, pooled_prompt_embeds, text_ids = pipe.encode_prompt(
            prompt=prompt, prompt_2=None, max_sequence_length=256
        )
    encoded_prompts.append({
        'prompt_embeds': prompt_embeds.cpu(),  # 移到CPU保存
        'pooled_prompt_embeds': pooled_prompt_embeds.cpu(),
        'text_ids': text_ids.cpu()
    })
    del prompt_embeds, pooled_prompt_embeds, text_ids
    torch.cuda.empty_cache()

print(f"   已编码 {len(encoded_prompts)} 个prompts")

# 现在可以安全卸载encoder了
print("🧹 卸载Text Encoders节省显存...")
del pipe.text_encoder, pipe.text_encoder_2, pipe.tokenizer, pipe.tokenizer_2
pipe.text_encoder = None
pipe.text_encoder_2 = None
pipe.tokenizer = None
pipe.tokenizer_2 = None
gc.collect()
torch.cuda.empty_cache()
pipe.to("cuda")

# ==========================================
# 3. 生成 FFT 密码本 W (所有图像共用)
# ==========================================
print("🔐 生成 FFT 密码本...")
secret_key = config['secret_key']
# 创建一个CUDA 上的随机数生成器
# 用 secret_key 作为固定随机种子，每次运行生成的随机数完全一样，可复现随机（Reproducible Randomness）
generator = torch.Generator(device='cuda').manual_seed(secret_key)
# 生成随机噪声 W_raw，形状为 (1, 1024, 64)，对应 32x32 的空间维度和 64 维的特征维度
W_raw = torch.randn((1, 1024, 64), generator=generator, device='cuda', dtype=torch.float32)
# 将 W_raw 重塑为 (1, 32, 32, 64) 以便进行空间频率分析
W_spatial = W_raw.view(1, 32, 32, 64)

# 对 W 进行 FFT 变换，得到频域表示 F_W，形状仍为 (1, 32, 32, 64)
F_W = torch.fft.fftshift(torch.fft.fft2(W_spatial, dim=(1, 2)), dim=(1, 2))
h, w = 32, 32
# 创建频率网格，计算每个位置到中心的距离 radius，用于构造带通滤波器
Y, X = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
# 设计一个带通滤波器 mask_fft，保留频率在 r_inner 和 r_outer 之间的成分，形状为 (1, 32, 32, 1)
center_y, center_x = h // 2, w // 2
# 计算每个位置到中心的距离 radius，形状为 (32, 32)，然后扩展为 (1, 32, 32, 1) 以便与 F_W 相乘
radius = torch.sqrt((Y - center_y)**2 + (X - center_x)**2).to('cuda')

r_inner = config['fft_radius_inner']
r_outer = config['fft_radius_outer']
# 构造带通滤波器 mask_fft，保留频率在 r_inner 和 r_outer 之间的成分，形状为 (1, 32, 32, 1)
mask_fft = ((radius >= r_inner) & (radius <= r_outer)).float().unsqueeze(0).unsqueeze(-1)

F_W_filtered = F_W * mask_fft
# 对过滤后的频域表示 F_W_filtered 进行逆 FFT 变换，得到空间域的密码本 W_filtered，形状为 (1, 32, 32, 64)
W_filtered = torch.fft.ifft2(torch.fft.ifftshift(F_W_filtered, dim=(1, 2)), dim=(1, 2)).real
W = W_filtered.view(1, 1024, 64).to(torch.bfloat16)

# Token 级能量均衡
W = W / (torch.norm(W, dim=-1, keepdim=True) + 1e-8)

# 语义掩码
M_spatial = torch.zeros((1, 32, 32, 1), device='cuda', dtype=torch.bfloat16)
# 根据配置指定的掩码区域，将 M_spatial 中对应位置设置为 1.0，表示这些位置是核心区域，将被注入水印
y_s, y_e, x_s, x_e = config['mask_region']
M_spatial[0, y_s:y_e, x_s:x_e, 0] = 1.0
M_attn = M_spatial.view(1, 1024, 1)
# 将掩码应用到密码本 W 上，得到最终的注入向量 W_anchored，只有核心区域的 token 会被注入水印
W_anchored = W * M_attn

# ==========================================
# 4. 正交注入回调函数
# ==========================================
alpha = config['alpha']

def orthogonal_drift_callback(pipe, step_index, timestep, callback_kwargs):
    latents = callback_kwargs["latents"]
    
    # Token-wise 正交化
    dot_W_x = torch.sum(W_anchored * latents, dim=-1, keepdim=True)
    dot_x_x = torch.sum(latents * latents, dim=-1, keepdim=True) + 1e-8
    # 计算正交分量 W_perp，确保注入的水印与当前的潜在表示正交，避免干扰生成过程中的语义信息
    W_perp = W_anchored - (dot_W_x / dot_x_x) * latents
    
    # 能量重归一化
    norm_W_anchored = torch.norm(W_anchored, dim=-1, keepdim=True)
    norm_W_perp = torch.norm(W_perp, dim=-1, keepdim=True) + 1e-8
    W_perp = W_perp * (norm_W_anchored / norm_W_perp)
    
    # 欧拉积分
    # 获取当前时间步的噪声尺度 sigmas，计算 dt，并根据欧拉方法更新 latents，注入正交水印分量 W_perp，注入强度由 alpha 控制
    # sigmas：扩散调度器的噪声强度序列，sigmas[step] 越大，当前步的噪声越多。
    # dt：相邻两步的噪声强度差，代表扩散步的 “时间增量”。
    # 加 abs() 是为了避免负步长导致水印方向反转
    # 目的：为什么要用 dt 加权？
    # 扩散模型前期噪声大（sigmas 大），此时加水印对图像影响小 → dt 会自动让前期注入更多水印。
    # 扩散模型后期噪声小（sigmas 小），此时加水印容易失真 → dt 会让后期注入更少水印。
    # 最终实现 “平滑嵌入”，水印既藏得深，又不破坏图像。
    sigmas = pipe.scheduler.sigmas
    dt = sigmas[step_index + 1] - sigmas[step_index]
    latents = latents + alpha * W_perp * abs(dt)
    
    callback_kwargs["latents"] = latents
    return callback_kwargs

# ==========================================
# 5. 批量生成主循环
# ==========================================
print(f"\n🎨 开始生成 {num_images} 张水印图像...\n")

results_summary = {
    'experiment_name': experiment_name,
    'timestamp': datetime.now().isoformat(),
    'total_images': num_images,
    'config': config,
    'images': []
}

for idx, prompt in enumerate(tqdm(prompts, desc="生成进度")):
    img_id = f"img_{idx:03d}"
    
    # 使用预编码的prompts
    encoded = encoded_prompts[idx]
    prompt_embeds = encoded['prompt_embeds'].to("cuda")
    pooled_prompt_embeds = encoded['pooled_prompt_embeds'].to("cuda")
    text_ids = encoded['text_ids'].to("cuda")
    
    # 生成图像
    image_out = pipe(
        prompt_embeds=prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        num_inference_steps=config['num_inference_steps'],
        guidance_scale=config['guidance_scale'],
        height=config['height'],
        width=config['width'],
        callback_on_step_end=orthogonal_drift_callback,
    ).images[0]
    
    # 保存图像
    img_filename = f"{img_id}.png"
    img_path = os.path.join(output_dir, img_filename)
    image_out.save(img_path)
    
    # ==========================================
    # 6. 提取差分签名
    # ==========================================
    img_tensor = T.ToTensor()(image_out).unsqueeze(0).to("cuda", dtype=torch.bfloat16)
    # 将像素值从 [0, 1] 线性映射到 [-1, 1]，以匹配 VAE 的输入范围
    img_tensor = (img_tensor - 0.5) * 2.0
    
    with torch.no_grad():
        # 将图像编码为潜在表示 z_0，使用 VAE 的编码器部分，得到形状为 (1, 1024, 64) 的潜在向量
        z_0 = pipe.vae.encode(img_tensor).latent_dist.sample()
        # 对 z_0 进行预处理，调整尺度和格式，以便输入到 Transformer 中进行签名提取
        z_0 = (z_0 - pipe.vae.config.shift_factor) * pipe.vae.config.scaling_factor
        # 将 z_0 从 (1, 64, 32, 32) 重塑为 (1, 1024, 64)，对应 32x32 的空间维度和 64 维的特征维度
        z_0 = pipe._pack_latents(z_0, batch_size=1, num_channels_latents=16, height=64, width=64)
        
        h = w = 32
        # 构造 img_ids，包含每个 token 的空间位置信息，形状为 (1, 1024, 3)，其中最后一个维度的三个通道分别编码了 token 的索引、y 坐标和 x 坐标的归一化值
        img_ids = torch.zeros(h, w, 3, device='cuda', dtype=torch.bfloat16)
        img_ids[..., 1] = torch.arange(h, device='cuda', dtype=torch.bfloat16).unsqueeze(1) / max(h - 1, 1)
        img_ids[..., 2] = torch.arange(w, device='cuda', dtype=torch.bfloat16).unsqueeze(0) / max(w - 1, 1)
        img_ids = img_ids.view(1, h * w, 3)
        
        # 在最后一个时间步（t_detect）使用 Transformer 提取潜在表示 z_0 中的特征，结合文本提示的嵌入信息，计算得到 v_pred，形状为 (1, 1024, 64)，包含了图像的语义信息和空间位置信息
        t_detect = torch.tensor([pipe.scheduler.sigmas[-1]], device="cuda", dtype=torch.bfloat16)
        v_pred = pipe.transformer(
            hidden_states=z_0, timestep=t_detect / 1000,
            pooled_projections=pooled_prompt_embeds,
            encoder_hidden_states=prompt_embeds,
            txt_ids=text_ids,
            img_ids=img_ids,
            return_dict=False,
        )[0]
    
    # 计算 8x8 签名
    v_pred_spatial = v_pred.view(32, 32, 64)
    W_anchored_spatial = W_anchored.view(32, 32, 64)
    S_orig = np.zeros((8, 8))
    
    for i in range(8):
        for j in range(8):
            W_patch = W_anchored_spatial[i*4:(i+1)*4, j*4:(j+1)*4, :].flatten().float()
            v_patch = v_pred_spatial[i*4:(i+1)*4, j*4:(j+1)*4, :].flatten().float()
            S_orig[i, j] = torch.nn.functional.cosine_similarity(
                W_patch.unsqueeze(0), v_patch.unsqueeze(0)
            ).item()
    
    # 保存签名
    sig_filename = f"{img_id}_signature.npy"
    sig_path = os.path.join(output_dir, sig_filename)
    np.save(sig_path, S_orig)
    
    # 核心区域统计
    mask_8x8 = np.zeros((8, 8))
    mask_8x8[1:7, 1:7] = 1.0
    # 计算核心区域（mask_8x8 == 1.0）对应的签名分数，统计核心区域的均值、标准差、最小值和最大值，保存到实验摘要中
    core_scores = S_orig[mask_8x8 == 1.0]
    
    img_result = {
        'id': img_id,
        'prompt': prompt,
        'image_file': img_filename,
        'signature_file': sig_filename,
        'stats': {
            'mean': float(np.mean(core_scores)),
            'std': float(np.std(core_scores)),
            'min': float(np.min(core_scores)),
            'max': float(np.max(core_scores))
        }
    }
    results_summary['images'].append(img_result)
    
    # 清理 GPU 缓存
    del prompt_embeds, pooled_prompt_embeds, text_ids
    torch.cuda.empty_cache()

# ==========================================
# 7. 保存实验摘要
# ==========================================
summary_path = os.path.join(output_dir, f'{experiment_name}_summary.json')
with open(summary_path, 'w') as f:
    json.dump(results_summary, f, indent=2)

print(f"\n✅ 批量生成完成！")
print(f"   图像保存位置: {output_dir}")
print(f"   实验摘要: {summary_path}")
print(f"\n📊 核心签名均值统计:")
means = [img['stats']['mean'] for img in results_summary['images']]
print(f"   平均值: {np.mean(means):.4f}")
print(f"   标准差: {np.std(means):.4f}")
print(f"   最小值: {np.min(means):.4f}")
print(f"   最大值: {np.max(means):.4f}")

print("\n✅ 批量生成完成！")
