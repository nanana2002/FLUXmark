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
# 1. 极速加载模型 (只保留检测需要的部件)
# ==========================================
print("正在加载检测器模型...")
pipe = FluxPipeline.from_pretrained(
    '/data/daiyina/project_flux/model/flux-schnell',
    torch_dtype=torch.bfloat16
)
# 我们做检测根本不需要算 Prompt 特征！我们用空文本特征 (Unconditional) 即可！
# 所以直接暴力删掉 Text Encoder，省下巨量显存！
# del pipe.text_encoder
# del pipe.text_encoder_2
# del pipe.tokenizer
# del pipe.tokenizer_2

pipe.text_encoder = None
pipe.text_encoder_2 = None
pipe.tokenizer = None
pipe.tokenizer_2 = None
gc.collect()
torch.cuda.empty_cache()

pipe.to("cuda")
print("✅ 检测器就绪！")

# ==========================================
# 2. 准备空文本特征 (用于无条件逆推)
# ==========================================
# FLUX 的文本特征维度是固定的，我们直接造一组全零的 Dummy 特征
# 这样不仅省去了文本编码，还能实现“盲提取”(不需要知道原图的 Prompt！)
batch_size = 1
prompt_embeds = torch.zeros((batch_size, 256, 4096), device="cuda", dtype=torch.bfloat16)
pooled_prompt_embeds = torch.zeros((batch_size, 768), device="cuda", dtype=torch.bfloat16)
text_ids = torch.zeros((256, 3), device="cuda", dtype=torch.bfloat16)

# ==========================================
# 3. 加载图片与生成密钥
# ==========================================
# 👉 请确保你在这里填入你【画了黑块】的图片路径！
image_path = '/data/daiyina/project_flux/pic/cat_with_box.png'
print(f"正在分析图片: {image_path}")

image = Image.open(image_path).convert("RGB")
img_tensor = T.ToTensor()(image).unsqueeze(0).to("cuda", dtype=torch.bfloat16)
img_tensor = (img_tensor - 0.5) * 2.0  # 归一化到 [-1, 1]

# 重新生成相同的密码本 W
secret_key = 42
generator = torch.Generator(device='cuda').manual_seed(secret_key)
latent_shape = (1, 1024, 64)
W = torch.randn(latent_shape, generator=generator, device='cuda', dtype=torch.bfloat16)

# ==========================================
# 4. 逆向提取残差速度
# ==========================================
with torch.no_grad():
    # VAE 编码
    z_0 = pipe.vae.encode(img_tensor).latent_dist.sample()
    z_0 = (z_0 - pipe.vae.config.shift_factor) * pipe.vae.config.scaling_factor
    z_0 = pipe._pack_latents(z_0, batch_size=1, num_channels_latents=16, height=64, width=64)

    # 重构 img_ids (你的高超操作！)
    h = w = 32
    img_ids = torch.zeros(h, w, 3, device='cuda', dtype=torch.bfloat16)
    img_ids[..., 1] = torch.arange(h, device='cuda', dtype=torch.bfloat16).unsqueeze(1) / max(h - 1, 1)
    img_ids[..., 2] = torch.arange(w, device='cuda', dtype=torch.bfloat16).unsqueeze(0) / max(w - 1, 1)
    img_ids = img_ids.view(h * w, 3)

    # 单步逆向推断，获取速度场
    t_detect = torch.tensor([pipe.scheduler.sigmas[-1]], device="cuda", dtype=torch.bfloat16)
    v_pred = pipe.transformer(
        hidden_states=z_0,
        timestep=t_detect / 1000,
        pooled_projections=pooled_prompt_embeds,
        encoder_hidden_states=prompt_embeds,
        txt_ids=text_ids,
        img_ids=img_ids,
        return_dict=False,
    )[0]

# ==========================================
# 5. 分块比对与 Z-Score 异常检测热力图
# ==========================================
grid_size = 32
W_spatial = W.view(grid_size, grid_size, 64)
v_pred_spatial = v_pred.view(grid_size, grid_size, 64)

patch_size = 4
heatmap_size = grid_size // patch_size
raw_heatmap = np.zeros((heatmap_size, heatmap_size))

# 提取各 Patch 的绝对相似度
for i in range(heatmap_size):
    for j in range(heatmap_size):
        h_start, h_end = i * patch_size, (i + 1) * patch_size
        w_start, w_end = j * patch_size, (j + 1) * patch_size
        
        W_patch = W_spatial[h_start:h_end, w_start:w_end, :].flatten().float()
        v_patch = v_pred_spatial[h_start:h_end, w_start:w_end, :].flatten().float()
        
        patch_sim = torch.nn.functional.cosine_similarity(W_patch.unsqueeze(0), v_patch.unsqueeze(0)).item()
        raw_heatmap[i, j] = patch_sim

# 计算全局均值和标准差
mean_sim = np.mean(raw_heatmap)
std_sim = np.std(raw_heatmap) + 1e-6 # 防止除零

print(f"\n🎯 [全局版权验证] 绝对平均相似度: {mean_sim:.4f}")

# 🚀 核心创新：计算 Z-Score 异常热力图
# Z-Score 反映了该区域偏离全局平均特征的程度
z_score_heatmap = (raw_heatmap - mean_sim) / std_sim

plt.figure(figsize=(6, 5))
# 对于 Z-Score，我们使用冷暖色调 (蓝-白-红)，经典的科学图表配色 (bwr)
# 蓝色代表正常（在平均线以上），红色代表异常/被篡改（严重低于平均线）
# 设置 vmin=-3, vmax=1 (负 3 个标准差绝对是极其严重的篡改)
plt.imshow(z_score_heatmap, cmap='bwr_r', interpolation='nearest', vmin=-3.0, vmax=1.0)
plt.colorbar(label='Tampering Z-Score (Anomaly)')
plt.title('Z-Score Tamper Localization (FlowMark)')
plt.axis('off')

heatmap_path = '/data/daiyina/project_flux/pic/heatmap_tampered.png'
plt.savefig(heatmap_path, bbox_inches='tight', dpi=300)
print(f"🚨 统计学级篡改热力图已生成！快去查看: {heatmap_path}")