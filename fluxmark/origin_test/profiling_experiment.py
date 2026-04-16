import torch
from diffusers import FluxPipeline
import torchvision.transforms as T
from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt
import os
import gc

os.environ['HF_HOME'] = '/data/daiyina/hf_cache'

# ==========================================
# 1. 加载模型与准备特征 (同前)
# ==========================================
pipe = FluxPipeline.from_pretrained('/data/daiyina/project_flux/model/flux-schnell', torch_dtype=torch.bfloat16)
del pipe.text_encoder, pipe.text_encoder_2, pipe.tokenizer, pipe.tokenizer_2
gc.collect()
torch.cuda.empty_cache()
pipe.to("cuda")

prompt_embeds = torch.zeros((1, 256, 4096), device="cuda", dtype=torch.bfloat16)
pooled_prompt_embeds = torch.zeros((1, 768), device="cuda", dtype=torch.bfloat16)
text_ids = torch.zeros((256, 3), device="cuda", dtype=torch.bfloat16)

secret_key = 42
generator = torch.Generator(device='cuda').manual_seed(secret_key)
W = torch.randn((1, 1024, 64), generator=generator, device='cuda', dtype=torch.bfloat16)
W_spatial = W.view(32, 32, 64)

# 我们直接定义一个提取 8x8 热力图的函数
def get_heatmap_for_image(img_tensor):
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
            hidden_states=z_0,
            timestep=torch.tensor([pipe.scheduler.sigmas[-1]], device="cuda", dtype=torch.bfloat16) / 1000,
            pooled_projections=pooled_prompt_embeds,
            encoder_hidden_states=prompt_embeds,
            txt_ids=text_ids,
            img_ids=img_ids,
            return_dict=False,
        )[0]

    v_pred_spatial = v_pred.view(32, 32, 64)
    heatmap = np.zeros((8, 8))
    for i in range(8):
        for j in range(8):
            W_patch = W_spatial[i*4:(i+1)*4, j*4:(j+1)*4, :].flatten().float()
            v_patch = v_pred_spatial[i*4:(i+1)*4, j*4:(j+1)*4, :].flatten().float()
            heatmap[i, j] = torch.nn.functional.cosine_similarity(W_patch.unsqueeze(0), v_patch.unsqueeze(0)).item()
    return heatmap

# ==========================================
# 2. 滑动窗口破坏实验 (The Sliding Window Vulnerability Test)
# ==========================================
original_img_path = '/data/daiyina/project_flux/pic/watermarked_cat.png'
orig_image = Image.open(original_img_path).convert("RGB")

print("正在提取原图基准特征...")
orig_tensor = T.ToTensor()(orig_image).unsqueeze(0).to("cuda", dtype=torch.bfloat16)
orig_tensor = (orig_tensor - 0.5) * 2.0
base_heatmap = get_heatmap_for_image(orig_tensor)

sensitivity_map = np.zeros((8, 8))
box_size = 512 // 8  # 每次遮挡 64x64 像素 (正好对应 8x8 热力图里的一个格子)

print("🚀 开始全图滑动探测敏感度 (这大概需要几分钟，总共测 64 次)...")
for i in range(8):
    for j in range(8):
        # 复制一张原图
        test_img = orig_image.copy()
        draw = ImageDraw.Draw(test_img)
        # 在 (j, i) 的位置画一个纯黑方块
        x0, y0 = j * box_size, i * box_size
        x1, y1 = x0 + box_size, y0 + box_size
        draw.rectangle([x0, y0, x1, y1], fill="black")
        
        # 转换并检测
        test_tensor = T.ToTensor()(test_img).unsqueeze(0).to("cuda", dtype=torch.bfloat16)
        test_tensor = (test_tensor - 0.5) * 2.0
        tamp_heatmap = get_heatmap_for_image(test_tensor)
        
        # 计算差分：只看[i, j] 这个被遮挡位置的余弦下降了多少！
        # 下降得越多，说明这里的水印越敏感
        delta = base_heatmap[i, j] - tamp_heatmap[i, j]
        sensitivity_map[i, j] = max(0, delta)  # 截断负数误差
        
        print(f"探测块 [{i},{j}] 完成，敏感度: {delta:.4f}")

# ==========================================
# 3. 绘制敏感度剖面图
# ==========================================
plt.figure(figsize=(6, 5))
# 用 'viridis' 色系：黄色/亮色代表极度敏感（背景），深蓝色代表迟钝（猫脸）
plt.imshow(sensitivity_map, cmap='viridis', interpolation='nearest')
plt.colorbar(label='Watermark Sensitivity ($\Delta$ Cosine Drop)')
plt.title('Spatial Vulnerability Profiling (Constant $\\alpha$)')
plt.axis('off')

save_path = '/data/daiyina/project_flux/pic/sensitivity_profiling.png'
plt.savefig(save_path, bbox_inches='tight', dpi=300)
print(f"\n 敏感度摸底图已生成！快去查看: {save_path}")