import torch
from diffusers import FluxPipeline
import torchvision.transforms as T
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
import gc

os.environ['HF_HOME'] = '/data/daiyina/hf_cache'

print("正在加载检测器模型...")
pipe = FluxPipeline.from_pretrained(
    '/data/daiyina/project_flux/model/flux-schnell',
    torch_dtype=torch.bfloat16
)
# del pipe.text_encoder, pipe.text_encoder_2, pipe.tokenizer, pipe.tokenizer_2
pipe.text_encoder = None
pipe.text_encoder_2 = None
pipe.tokenizer = None
pipe.tokenizer_2 = None

gc.collect()
torch.cuda.empty_cache()
pipe.to("cuda")
print("✅ 检测器就绪！")

# 准备空文本特征
prompt_embeds = torch.zeros((1, 256, 4096), device="cuda", dtype=torch.bfloat16)
pooled_prompt_embeds = torch.zeros((1, 768), device="cuda", dtype=torch.bfloat16)
text_ids = torch.zeros((256, 3), device="cuda", dtype=torch.bfloat16)

# 统一的密钥生成
secret_key = 42
generator = torch.Generator(device='cuda').manual_seed(secret_key)
latent_shape = (1, 1024, 64)
W = torch.randn(latent_shape, generator=generator, device='cuda', dtype=torch.bfloat16)
W_spatial = W.view(32, 32, 64)

def get_raw_heatmap(image_path):
    """提取单张图片的 8x8 原始余弦相似度矩阵"""
    print(f"正在分析: {image_path}")
    image = Image.open(image_path).convert("RGB")
    img_tensor = T.ToTensor()(image).unsqueeze(0).to("cuda", dtype=torch.bfloat16)
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
            hidden_states=z_0,
            timestep=t_detect / 1000,
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
# 核心逻辑：执行差分运算 (Differential Analysis)
# ==========================================
# 👉 这里填入你之前生成的纯净带水印的猫
original_img_path = '/data/daiyina/project_flux/pic/watermarked_cat.png' 
# 👉 这里填入你画了黑块的猫 (比如左上角或者嘴巴有黑块的那张)
tampered_img_path = '/data/daiyina/project_flux/pic/cat_with_box5.png' 

print("\n--- 提取原始签名 ---")
heatmap_orig = get_raw_heatmap(original_img_path)

print("\n--- 提取篡改特征 ---")
heatmap_tamp = get_raw_heatmap(tampered_img_path)

# 💡 你的天才公式：差分热力图！
# 正常区域：差值接近 0。被篡改区域（原本是正数，篡改后变成负数或0）：差值会变大！
delta_heatmap = heatmap_orig - heatmap_tamp

# ==========================================
# 绘制绝杀对比图
# ==========================================
# 为了让论文里的图极其漂亮，我们把负差值（比如正常波动的白噪音）截断在 0，只显示正差值（被破坏的区域）
delta_heatmap_clean = np.clip(delta_heatmap, a_min=0, a_max=None)

plt.figure(figsize=(6, 5))
# 使用 'Reds' 单色阶：0（未篡改）是纯白色，数值越大（篡改越严重）颜色越血红！
plt.imshow(delta_heatmap_clean, cmap='Reds', interpolation='nearest', vmin=0.0, vmax=0.15)
plt.colorbar(label='Tampering Severity ($\Delta$ Cosine Similarity)')
plt.title('Differential Tamper Localization (FlowMark)')
plt.axis('off')

save_path = '/data/daiyina/project_flux/pic/heatmap_differential.png'
plt.savefig(save_path, bbox_inches='tight', dpi=300)
print(f"\n🚨 绝杀级差分热力图已生成！快去查看: {save_path}")