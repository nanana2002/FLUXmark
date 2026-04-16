import torch
from diffusers import FluxPipeline
import torchvision.transforms as T
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
import gc

os.environ['HF_HOME'] = '/data/daiyina/hf_cache'

# ==========================================
# 1. 极致加载模型 (无文本编码器以省显存)
# ==========================================
print("🚀 [Step 1] 正在加载 FLUX 流匹配模型...")
pipe = FluxPipeline.from_pretrained('/data/daiyina/project_flux/model/flux-schnell', torch_dtype=torch.bfloat16)

# 我们直接使用空文本特征进行无条件生成和检测 (绝对的 Blind!)
batch_size = 1
prompt_embeds = torch.zeros((batch_size, 256, 4096), device="cuda", dtype=torch.bfloat16)
pooled_prompt_embeds = torch.zeros((batch_size, 768), device="cuda", dtype=torch.bfloat16)
text_ids = torch.zeros((256, 3), device="cuda", dtype=torch.bfloat16)

# del pipe.text_encoder, pipe.text_encoder_2, pipe.tokenizer, pipe.tokenizer_2
pipe.text_encoder = None
pipe.text_encoder_2 = None
pipe.tokenizer = None
pipe.tokenizer_2 = None
gc.collect()
torch.cuda.empty_cache()
pipe.to("cuda")

# ==========================================
# 2. 频域约束水印生成 (FFT Mid-frequency Bandpass)
# ==========================================
print("🌊[Step 2] 正在生成 FFT 中频密码本...")
secret_key = 42
generator = torch.Generator(device='cuda').manual_seed(secret_key)
W_raw = torch.randn((1, 1024, 64), generator=generator, device='cuda', dtype=torch.float32) # FFT需float32
W_spatial = W_raw.view(1, 32, 32, 64)

# 执行 2D 傅里叶变换
F_W = torch.fft.fftshift(torch.fft.fft2(W_spatial, dim=(1, 2)), dim=(1, 2))

# 创建环形带通滤波器 (保留中频段)
h, w = 32, 32
Y, X = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
center_y, center_x = h // 2, w // 2
radius = torch.sqrt((Y - center_y)**2 + (X - center_x)**2).to('cuda')
mask = ((radius >= 4) & (radius <= 12)).float().unsqueeze(0).unsqueeze(-1) # 剔除极低频(0-4)和极高频(12-16)

# 应用掩码并逆变换
F_W_filtered = F_W * mask
W_filtered = torch.fft.ifft2(torch.fft.ifftshift(F_W_filtered, dim=(1, 2)), dim=(1, 2)).real
W = W_filtered.view(1, 1024, 64).to(torch.bfloat16) # 转回 bfloat16

# ==========================================
# 3. 语义锚定掩码 (Semantic Anchoring Mask)
# ==========================================
# 为了保持代码简洁且不依赖外部大模型，我们在这里生成一个中心区域的 32x32 Mask 模拟 "猫" 的主体区域。
# 在真实论文实验中，这可以替换为从 MMDiT 提取的 Attention Map 或 CLIPSeg 掩码。
M_spatial = torch.zeros((1, 32, 32, 1), device='cuda', dtype=torch.bfloat16)
M_spatial[0, 6:26, 6:26, 0] = 1.0  # 假设图像中央 20x20 的潜变量网格是猫的主体
M = M_spatial.view(1, 1024, 1)

W_anchored = W * M  # 基因锁：水印只在掩码为 1 的地方存在！

# ==========================================
# 4. 正交流形注入 Callback (Orthogonal Drift)
# ==========================================
alpha = 0.25 # 因为正交零干扰，我们可以将注入强度提得很高！

def orthogonal_drift_callback(pipe, step_index, timestep, callback_kwargs):
    latents = callback_kwargs["latents"]
    
    # 🌟 核心：Gram-Schmidt 正交化
    # 确保 W_anchored 垂直于当前的潜变量生成流形 latents
    dot_W_x = torch.sum(W_anchored * latents)
    dot_x_x = torch.sum(latents * latents) + 1e-8
    
    W_perp = W_anchored - (dot_W_x / dot_x_x) * latents
    
    # 能量重归一化，保证强度不衰减
    W_perp = W_perp * (torch.norm(W_anchored) / (torch.norm(W_perp) + 1e-8))
    
    sigmas = pipe.scheduler.sigmas
    dt = sigmas[step_index + 1] - sigmas[step_index] 
    
    # 将正交分量无损注入！
    latents = latents + alpha * W_perp * abs(dt)
    
    callback_kwargs["latents"] = latents
    return callback_kwargs

print("🎨 [Step 3] 正在使用正交流形魔法生成图像...")
image_out = pipe(
    prompt_embeds=prompt_embeds,
    pooled_prompt_embeds=pooled_prompt_embeds,
    num_inference_steps=4,
    guidance_scale=0.0,
    height=512, width=512,
    callback_on_step_end=orthogonal_drift_callback, 
).images[0]

img_path = '/data/daiyina/project_flux/pic/ortho_watermarked.png'
image_out.save(img_path)
print(f"✅ 正交水印图已保存至: {img_path}")

# ==========================================
# 5. 盲提取与自适应篡改定位 (Blind & Self-Adaptive)
# ==========================================
print("🔍 [Step 4] 正在进行单步盲提取...")
img_tensor = T.ToTensor()(image_out).unsqueeze(0).to("cuda", dtype=torch.bfloat16)
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

    t_detect = torch.tensor([pipe.scheduler.sigmas[-1]], device="cuda", dtype=torch.bfloat16)
    v_pred = pipe.transformer(
        hidden_states=z_0, timestep=t_detect / 1000,
        pooled_projections=pooled_prompt_embeds, encoder_hidden_states=prompt_embeds,
        txt_ids=text_ids, img_ids=img_ids, return_dict=False,
    )[0]

# ==========================================
# 6. 计算绝对均匀的热力图 (The Flat Mirror)
# ==========================================
v_pred_spatial = v_pred.view(32, 32, 64)
W_anchored_spatial = W_anchored.view(32, 32, 64)
heatmap = np.zeros((8, 8))

# 收集锚定区域内所有的余弦值，用来算自适应阈值
anchored_scores =[]

for i in range(8):
    for j in range(8):
        W_patch = W_anchored_spatial[i*4:(i+1)*4, j*4:(j+1)*4, :].flatten().float()
        v_patch = v_pred_spatial[i*4:(i+1)*4, j*4:(j+1)*4, :].flatten().float()
        
        sim = torch.nn.functional.cosine_similarity(W_patch.unsqueeze(0), v_patch.unsqueeze(0)).item()
        heatmap[i, j] = sim
        
        # 只统计主体区域（我们之前设定的中心 20x20，大约在 8x8 热力图的 [1:6, 1:6] 范围内）
        if 1 <= i <= 6 and 1 <= j <= 6:
            anchored_scores.append(sim)

# 🌟 内部自适应阈值！完全不需要原图！
print(anchored_scores)
mean_score = np.mean(anchored_scores)
print(f"🎯 主体区域的平均相似度 (平坦镜面值): {mean_score:.4f}")

plt.figure(figsize=(6, 5))
# 用热力图展示这面“完美的镜子”，你应该看到中心区域是一片极其均匀的暖色（分数很高），四周是 0
plt.imshow(heatmap, cmap='YlOrRd', interpolation='nearest', vmin=0.0, vmax=mean_score + 0.05)
plt.colorbar(label='Cosine Similarity (Blind Extraction)')
plt.title(f'OrthoFlow Uniform Heatmap (Mean: {mean_score:.3f})')
plt.axis('off')

heatmap_path = '/data/daiyina/project_flux/pic/ortho_heatmap_clean.png'
plt.savefig(heatmap_path, bbox_inches='tight', dpi=300)
print(f"🚨 [成功] 完美平滑的原图热力图已生成: {heatmap_path}")