import torch
from diffusers import FluxPipeline
import torchvision.transforms as T
from torchvision.transforms import functional as F_vision
from PIL import Image
import numpy as np
import io
import os
import gc

os.environ['HF_HOME'] = '/data/daiyina/hf_cache'
output_dir = '/data/daiyina/project_flux/pic'

# ==========================================
# 1. 极致加载模型 (检测专用)
# ==========================================
print("🚀 [1] 正在加载检测器...")
pipe = FluxPipeline.from_pretrained('/data/daiyina/project_flux/model/flux-schnell', torch_dtype=torch.bfloat16)
batch_size = 1
prompt_embeds = torch.zeros((batch_size, 256, 4096), device="cuda", dtype=torch.bfloat16)
pooled_prompt_embeds = torch.zeros((batch_size, 768), device="cuda", dtype=torch.bfloat16)
text_ids = torch.zeros((256, 3), device="cuda", dtype=torch.bfloat16)
pipe.text_encoder = None
pipe.text_encoder_2 = None
pipe.tokenizer = None
pipe.tokenizer_2 = None
gc.collect()
torch.cuda.empty_cache()
pipe.to("cuda")

# 重建相同的 W (必须有才能测)
secret_key = 42
generator = torch.Generator(device='cuda').manual_seed(secret_key)
W_raw = torch.randn((1, 1024, 64), generator=generator, device='cuda', dtype=torch.float32)
W_spatial = W_raw.view(1, 32, 32, 64)
F_W = torch.fft.fftshift(torch.fft.fft2(W_spatial, dim=(1, 2)), dim=(1, 2))
h, w = 32, 32
Y, X = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
center_y, center_x = h // 2, w // 2
radius = torch.sqrt((Y - center_y)**2 + (X - center_x)**2).to('cuda')
mask_fft = ((radius >= 4) & (radius <= 12)).float().unsqueeze(0).unsqueeze(-1) 
F_W_filtered = F_W * mask_fft
W_filtered = torch.fft.ifft2(torch.fft.ifftshift(F_W_filtered, dim=(1, 2)), dim=(1, 2)).real
W = W_filtered.view(1, 1024, 64).to(torch.bfloat16)

M_spatial = torch.zeros((1, 32, 32, 1), device='cuda', dtype=torch.bfloat16)
M_spatial[0, 6:26, 6:26, 0] = 1.0 
M = M_spatial.view(1, 1024, 1)
W_anchored = W * M

# ==========================================
# 2. 读取你说的“真实基准” (S_orig)
# ==========================================
signature_path = os.path.join(output_dir, 'signature_orig.npy')
S_orig = np.load(signature_path)

# 我们只统计核心锚定区域 (Mask为1的中心区域) 的真实基准均值
# core_mask = M_spatial.view(32, 32).cpu().numpy()
core_mask = M_spatial.view(32, 32).cpu().float().numpy()
# 8x8 热力图中，对应的核心区域大概是 [1:7, 1:7]
mask_8x8 = np.zeros((8, 8))
mask_8x8[1:7, 1:7] = 1.0

# 计算包含提取误差的真实基准均值
score_baseline = np.mean(S_orig[mask_8x8 == 1.0])
print(f"🎯 [基准档案] 从 signature_orig.npy 中读取的纯净水印分数: {score_baseline:.4f}")

# ==========================================
# 3. 提取受攻击图像的 S_tamp 函数
# ==========================================
def extract_S_tamp(pil_image):
    img_tensor = T.ToTensor()(pil_image).unsqueeze(0).to("cuda", dtype=torch.bfloat16)
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
            hidden_states=z_0, timestep=torch.tensor([pipe.scheduler.sigmas[-1]], device="cuda", dtype=torch.bfloat16) / 1000,
            pooled_projections=pooled_prompt_embeds, encoder_hidden_states=prompt_embeds,
            txt_ids=text_ids, img_ids=img_ids, return_dict=False,
        )[0]
    
    v_pred_spatial = v_pred.view(32, 32, 64)
    W_anchored_spatial = W_anchored.view(32, 32, 64)
    S_tamp = np.zeros((8, 8))
    for i in range(8):
        for j in range(8):
            W_patch = W_anchored_spatial[i*4:(i+1)*4, j*4:(j+1)*4, :].flatten().float()
            v_patch = v_pred_spatial[i*4:(i+1)*4, j*4:(j+1)*4, :].flatten().float()
            S_tamp[i, j] = torch.nn.functional.cosine_similarity(W_patch.unsqueeze(0), v_patch.unsqueeze(0)).item()
            
    # 返回受攻击图像在核心区域的绝对均值
    return np.mean(S_tamp[mask_8x8 == 1.0])

# ==========================================
# 4. 执行严酷的鲁棒性攻击
# ==========================================
original_image_path = os.path.join(output_dir, 'orthoflow_final_cat.png')
orig_img = Image.open(original_image_path).convert("RGB")

print("\n💥 [攻击 1] JPEG 压缩 (Quality = 50, 模拟微信朋友圈渣画质)...")
buffer = io.BytesIO()
orig_img.save(buffer, format="JPEG", quality=50)
jpeg_img = Image.open(buffer)
score_jpeg = extract_S_tamp(jpeg_img)
print(f"   => 提取的绝对分数 S_tamp: {score_jpeg:.4f}")
print(f"   => 相对基准保留率: {score_jpeg / score_baseline * 100:.1f}%")
print(f"   => 版权裁定: {'✅ 胜诉 (远大于 0.02)' if score_jpeg > 0.02 else '❌ 败诉'}")

print("\n💥 [攻击 2] 高斯模糊 (Sigma = 2.0, 破坏高频纹理)...")
blur_img = F_vision.gaussian_blur(orig_img, kernel_size=[9, 9], sigma=[2.0, 2.0])
score_blur = extract_S_tamp(blur_img)
print(f"   => 提取的绝对分数 S_tamp: {score_blur:.4f}")
print(f"   => 相对基准保留率: {score_blur / score_baseline * 100:.1f}%")
print(f"   => 版权裁定: {'✅ 胜诉 (远大于 0.02)' if score_blur > 0.02 else '❌ 败诉'}")

print("\n🎉 实验闭环！你可以理直气壮地把这些保留率数据填进 Table 1 了！")