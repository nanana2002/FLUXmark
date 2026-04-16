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
# 2. 读取生成时保存的"黄金标准"签名 S_orig
# ==========================================
signature_path = os.path.join(output_dir, 'signature_orig.npy')
S_orig = np.load(signature_path)  # 形状 (8, 8)
print(f"🎯 [黄金标准] 已从 signature_orig.npy 加载原始签名，形状: {S_orig.shape}")

# 核心锚定区域掩码 (8x8 热力图中对应主体区域)
mask_8x8 = np.zeros((8, 8))
mask_8x8[1:7, 1:7] = 1.0  # 中心 6x6 区域

# ==========================================
# 3. 提取签名的函数
# ==========================================
def extract_signature(pil_image):
    """从图像中提取 8x8 签名矩阵 S"""
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
            hidden_states=z_0, timestep=torch.tensor([pipe.scheduler.sigmas[-1]], device="cuda", dtype=torch.bfloat16)
/ 1000,
            pooled_projections=pooled_prompt_embeds, encoder_hidden_states=prompt_embeds,
            txt_ids=text_ids, img_ids=img_ids, return_dict=False,
        )[0]

    v_pred_spatial = v_pred.view(32, 32, 64)
    W_anchored_spatial = W_anchored.view(32, 32, 64)
    S = np.zeros((8, 8))
    for i in range(8):
        for j in range(8):
            W_patch = W_anchored_spatial[i*4:(i+1)*4, j*4:(j+1)*4, :].flatten().float()
            v_patch = v_pred_spatial[i*4:(i+1)*4, j*4:(j+1)*4, :].flatten().float()
            S[i, j] = torch.nn.functional.cosine_similarity(W_patch.unsqueeze(0), v_patch.unsqueeze(0)).item()

    return S

def compare_signatures(S_orig, S_tamp, mask):
    """
    比较两个签名的相似度
    返回: (相似度分数, 相关系数, MSE)
    """
    # 只比较掩码区域
    S_orig_masked = S_orig[mask == 1.0]
    S_tamp_masked = S_tamp[mask == 1.0]

    # 1. 余弦相似度 (向量夹角)
    cos_sim = np.dot(S_orig_masked, S_tamp_masked) / (np.linalg.norm(S_orig_masked) * np.linalg.norm(S_tamp_masked) +
1e-8)

    # 2. 皮尔逊相关系数
    corr = np.corrcoef(S_orig_masked, S_tamp_masked)[0, 1]

    # 3. 均方误差 (越小越好)
    mse = np.mean((S_orig_masked - S_tamp_masked) ** 2)

    return cos_sim, corr, mse

# ==========================================
# 4. 从原始图像提取签名进行对比 (验证提取准确性)
# ==========================================
original_image_path = os.path.join(output_dir, 'orthoflow_final_cat.png')
orig_img = Image.open(original_image_path).convert("RGB")

print("\n🔍 [验证] 从原始水印图像提取签名，与保存的 S_orig 对比...")
S_extracted = extract_signature(orig_img)
cos_sim, corr, mse = compare_signatures(S_orig, S_extracted, mask_8x8)

print(f"   => 余弦相似度: {cos_sim:.6f} (越接近1越好)")
print(f"   => 皮尔逊相关系数: {corr:.6f} (越接近1越好)")
print(f"   => 均方误差(MSE): {mse:.8f} (越接近0越好)")

if cos_sim > 0.99 and corr > 0.99:
    print("   => ✅ 提取验证通过！提取的签名与保存的签名几乎一致")
else:
    print("   => ⚠️ 提取验证警告！签名存在差异，请检查提取过程")

# ==========================================
# 5. 执行鲁棒性攻击测试
# ==========================================

print("\n💥 [攻击 1] JPEG 压缩 (Quality = 50, 模拟微信朋友圈渣画质)...")
buffer = io.BytesIO()
orig_img.save(buffer, format="JPEG", quality=50)
jpeg_img = Image.open(buffer)
S_jpeg = extract_signature(jpeg_img)
cos_sim_jpeg, corr_jpeg, mse_jpeg = compare_signatures(S_orig, S_jpeg, mask_8x8)

print(f"   => 余弦相似度: {cos_sim_jpeg:.4f} (保留率: {cos_sim_jpeg/cos_sim*100:.1f}%)")
print(f"   => 皮尔逊相关系数: {corr_jpeg:.4f}")
print(f"   => MSE: {mse_jpeg:.6f}")
print(f"   => 版权裁定: {'✅ 水印可检测' if cos_sim_jpeg > 0.8 else '❌ 水印丢失'}")

print("\n💥 [攻击 2] JPEG 压缩 (Quality = 30, 更激进的压缩)...")
buffer = io.BytesIO()
orig_img.save(buffer, format="JPEG", quality=30)
jpeg_img_30 = Image.open(buffer)
S_jpeg_30 = extract_signature(jpeg_img_30)
cos_sim_jpeg_30, corr_jpeg_30, mse_jpeg_30 = compare_signatures(S_orig, S_jpeg_30, mask_8x8)
print(f"   => 余弦相似度: {cos_sim_jpeg_30:.4f}")
print(f"   => 版权裁定: {'✅ 水印可检测' if cos_sim_jpeg_30 > 0.8 else '❌ 水印可能丢失'}")

print("\n💥 [攻击 3] 高斯模糊 (Sigma = 2.0, 破坏高频纹理)...")
blur_img = F_vision.gaussian_blur(orig_img, kernel_size=[9, 9], sigma=[2.0, 2.0])
S_blur = extract_signature(blur_img)
cos_sim_blur, corr_blur, mse_blur = compare_signatures(S_orig, S_blur, mask_8x8)

print(f"   => 余弦相似度: {cos_sim_blur:.4f} (保留率: {cos_sim_blur/cos_sim*100:.1f}%)")
print(f"   => 皮尔逊相关系数: {corr_blur:.4f}")
print(f"   => MSE: {mse_blur:.6f}")
print(f"   => 版权裁定: {'✅ 水印可检测' if cos_sim_blur > 0.8 else '❌ 水印丢失'}")

print("\n💥 [攻击 4] 高斯模糊 (Sigma = 3.0, 更强的模糊)...")
blur_img_3 = F_vision.gaussian_blur(orig_img, kernel_size=[11, 11], sigma=[3.0, 3.0])
S_blur_3 = extract_signature(blur_img_3)
cos_sim_blur_3, _, _ = compare_signatures(S_orig, S_blur_3, mask_8x8)
print(f"   => 余弦相似度: {cos_sim_blur_3:.4f}")
print(f"   => 版权裁定: {'✅ 水印可检测' if cos_sim_blur_3 > 0.8 else '❌ 水印可能丢失'}")

print("\n🎉 实验完成！以下是总结表格数据:")
print("=" * 60)
print(f"{'攻击类型':<25} {'余弦相似度':<15} {'保留率':<10}")
print("-" * 60)
print(f"{'原始图像 (基准)':<25} {cos_sim:<15.4f} {'100.0%':<10}")
print(f"{'JPEG Q=50':<25} {cos_sim_jpeg:<15.4f} {cos_sim_jpeg/cos_sim*100:<10.1f}%")
print(f"{'JPEG Q=30':<25} {cos_sim_jpeg_30:<15.4f} {cos_sim_jpeg_30/cos_sim*100:<10.1f}%")
print(f"{'高斯模糊 σ=2.0':<25} {cos_sim_blur:<15.4f} {cos_sim_blur/cos_sim*100:<10.1f}%")
print(f"{'高斯模糊 σ=3.0':<25} {cos_sim_blur_3:<15.4f} {cos_sim_blur_3/cos_sim*100:<10.1f}%")
print("=" * 60)
