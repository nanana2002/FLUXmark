import torch
from diffusers import FluxPipeline
import torchvision.transforms as T
from torchvision.transforms import functional as F_vision
from PIL import Image
import numpy as np
import io
import os
import gc
import json

os.environ['HF_HOME'] = '/data/daiyina/hf_cache'
output_dir = '/data/daiyina/project_flux/pic'

# ==========================================
# 1. 加载配置和模型
# ==========================================
print("🚀 [1] 正在加载配置...")
with open(os.path.join(output_dir, 'watermark_config.json'), 'r') as f:
    config = json.load(f)

prompt = config['prompt']
secret_key = config['secret_key']
alpha = config['alpha']

print(f"   提示词: {prompt}")
print(f"   密钥: {secret_key}")

print("🚀 [2] 正在加载检测器...")
pipe = FluxPipeline.from_pretrained(
    '/data/daiyina/project_flux/model/flux-schnell', 
    torch_dtype=torch.bfloat16
)

# 使用相同的 prompt 进行编码！
print(">>> 正在编码 Prompt (与生成时相同)...")
with torch.no_grad():
    prompt_embeds, pooled_prompt_embeds, text_ids = pipe.encode_prompt(
        prompt=prompt, prompt_2=None, max_sequence_length=256
    )

# 移动到 CUDA
prompt_embeds = prompt_embeds.to("cuda")
pooled_prompt_embeds = pooled_prompt_embeds.to("cuda")
text_ids = text_ids.to("cuda")

# 卸载文本编码器
del pipe.text_encoder, pipe.text_encoder_2, pipe.tokenizer, pipe.tokenizer_2
pipe.text_encoder = None
pipe.text_encoder_2 = None
pipe.tokenizer = None
pipe.tokenizer_2 = None
gc.collect()
torch.cuda.empty_cache()
pipe.to("cuda")

# ==========================================
# 2. 重建密码本 W (必须与生成时完全一致)
# ==========================================
print("🚀 [3] 重建 FFT 密码本...")
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

# Token 级能量均衡
W = W / (torch.norm(W, dim=-1, keepdim=True) + 1e-8)

# 重建语义掩码
M_spatial = torch.zeros((1, 32, 32, 1), device='cuda', dtype=torch.bfloat16)
y_start, y_end, x_start, x_end = config['mask_region']
M_spatial[0, y_start:y_end, x_start:x_end, 0] = 1.0
M = M_spatial.view(1, 1024, 1)
W_anchored = W * M

# ==========================================
# 3. 加载黄金标准签名 S_orig
# ==========================================
signature_path = os.path.join(output_dir, 'signature_orig_fixed.npy')
S_orig = np.load(signature_path)
print(f"🎯 [黄金标准] 已加载 S_orig，形状: {S_orig.shape}")

# 核心锚定区域掩码 (8x8)
mask_8x8 = np.zeros((8, 8))
# 计算 32x32 到 8x8 的映射
y_start_8 = y_start // 4
y_end_8 = y_end // 4
x_start_8 = x_start // 4
x_end_8 = x_end // 4
mask_8x8[y_start_8:y_end_8, x_start_8:x_end_8] = 1.0
print(f"   核心区域 (8x8): y[{y_start_8}:{y_end_8}], x[{x_start_8}:{x_end_8}]")

# ==========================================
# 4. 提取签名函数
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

        # 使用与生成时相同的条件！
        v_pred = pipe.transformer(
            hidden_states=z_0, 
            timestep=torch.tensor([pipe.scheduler.sigmas[-1]], device="cuda", dtype=torch.bfloat16) / 1000,
            pooled_projections=pooled_prompt_embeds,  # 相同条件
            encoder_hidden_states=prompt_embeds,       # 相同条件
            txt_ids=text_ids,                          # 相同条件
            img_ids=img_ids, 
            return_dict=False,
        )[0]
    
    v_pred_spatial = v_pred.view(32, 32, 64)
    W_anchored_spatial = W_anchored.view(32, 32, 64)
    S = np.zeros((8, 8))
    
    for i in range(8):
        for j in range(8):
            W_patch = W_anchored_spatial[i*4:(i+1)*4, j*4:(j+1)*4, :].flatten().float()
            v_patch = v_pred_spatial[i*4:(i+1)*4, j*4:(j+1)*4, :].flatten().float()
            S[i, j] = torch.nn.functional.cosine_similarity(
                W_patch.unsqueeze(0), v_patch.unsqueeze(0)
            ).item()
    
    return S

def compare_signatures(S_orig, S_tamp, mask):
    """比较两个签名的相似度"""
    S_orig_masked = S_orig[mask == 1.0]
    S_tamp_masked = S_tamp[mask == 1.0]
    
    # 余弦相似度
    cos_sim = np.dot(S_orig_masked, S_tamp_masked) / (
        np.linalg.norm(S_orig_masked) * np.linalg.norm(S_tamp_masked) + 1e-8
    )
    
    # 皮尔逊相关系数
    if len(S_orig_masked) > 1:
        corr = np.corrcoef(S_orig_masked, S_tamp_masked)[0, 1]
    else:
        corr = 0.0
    
    # 均方误差
    mse = np.mean((S_orig_masked - S_tamp_masked) ** 2)
    
    # 绝对均值差异
    mean_diff = abs(np.mean(S_orig_masked) - np.mean(S_tamp_masked))
    
    return cos_sim, corr, mse, mean_diff

def print_signature_stats(S, mask, name):
    """打印签名统计信息"""
    scores = S[mask == 1.0]
    print(f"   {name} - 均值: {np.mean(scores):.4f}, 标准差: {np.std(scores):.4f}")

# ==========================================
# 5. 验证原始图像提取准确性
# ==========================================
original_image_path = os.path.join(output_dir, 'orthoflow_final_cat_fixed.png')
orig_img = Image.open(original_image_path).convert("RGB")

print("\n" + "="*60)
print("🔍 [验证阶段] 从原始图像提取签名，与保存的 S_orig 对比...")
print("="*60)

S_extracted = extract_signature(orig_img)
cos_sim, corr, mse, mean_diff = compare_signatures(S_orig, S_extracted, mask_8x8)

print_signature_stats(S_orig, mask_8x8, "保存的 S_orig")
print_signature_stats(S_extracted, mask_8x8, "提取的 S_extracted")

print(f"\n   余弦相似度: {cos_sim:.6f} (越接近1越好)")
print(f"   皮尔逊相关系数: {corr:.6f} (越接近1越好)")
print(f"   MSE: {mse:.8f} (越接近0越好)")
print(f"   均值差异: {mean_diff:.6f}")

if cos_sim > 0.99 and mse < 0.001:
    print("   ✅ 提取验证通过！签名完全一致")
else:
    print(f"   ⚠️ 提取验证警告！请检查实现")

# ==========================================
# 6. 鲁棒性攻击测试
# ==========================================
print("\n" + "="*60)
print("💥 [鲁棒性测试阶段]")
print("="*60)

results = []

# 测试 1: 原始图像
print("\n[测试 1] 原始图像 (无损)")
cos_sim_clean, corr_clean, mse_clean, _ = compare_signatures(S_orig, S_extracted, mask_8x8)
print(f"   余弦相似度: {cos_sim_clean:.4f}")
print(f"   版权判定: {'✅ 通过' if cos_sim_clean > 0.8 else '❌ 失败'}")
results.append(("原始图像", cos_sim_clean))

# 测试 2: JPEG Q=75
print("\n[测试 2] JPEG 压缩 (Quality = 75)")
buffer = io.BytesIO()
orig_img.save(buffer, format="JPEG", quality=75)
jpeg_75 = Image.open(buffer)
jpeg_75.save(os.path.join(output_dir, 'attack_jpeg_75.jpg'))
print(f"   已保存: attack_jpeg_75.jpg")
S_jpeg_75 = extract_signature(jpeg_75)
cos_sim_j75, _, _, _ = compare_signatures(S_orig, S_jpeg_75, mask_8x8)
print(f"   余弦相似度: {cos_sim_j75:.4f} (保留率: {cos_sim_j75/cos_sim_clean*100:.1f}%)")
print(f"   版权判定: {'✅ 通过' if cos_sim_j75 > 0.8 else '❌ 失败'}")
results.append(("JPEG Q=75", cos_sim_j75))

# 测试 3: JPEG Q=50
print("\n[测试 3] JPEG 压缩 (Quality = 50)")
buffer = io.BytesIO()
orig_img.save(buffer, format="JPEG", quality=50)
jpeg_50 = Image.open(buffer)
jpeg_50.save(os.path.join(output_dir, 'attack_jpeg_50.jpg'))
print(f"   已保存: attack_jpeg_50.jpg")
S_jpeg_50 = extract_signature(jpeg_50)
cos_sim_j50, _, _, _ = compare_signatures(S_orig, S_jpeg_50, mask_8x8)
print(f"   余弦相似度: {cos_sim_j50:.4f} (保留率: {cos_sim_j50/cos_sim_clean*100:.1f}%)")
print(f"   版权判定: {'✅ 通过' if cos_sim_j50 > 0.8 else '❌ 失败'}")
results.append(("JPEG Q=50", cos_sim_j50))

# 测试 4: JPEG Q=30
print("\n[测试 4] JPEG 压缩 (Quality = 30)")
buffer = io.BytesIO()
orig_img.save(buffer, format="JPEG", quality=30)
jpeg_30 = Image.open(buffer)
jpeg_30.save(os.path.join(output_dir, 'attack_jpeg_30.jpg'))
print(f"   已保存: attack_jpeg_30.jpg")
S_jpeg_30 = extract_signature(jpeg_30)
cos_sim_j30, _, _, _ = compare_signatures(S_orig, S_jpeg_30, mask_8x8)
print(f"   余弦相似度: {cos_sim_j30:.4f} (保留率: {cos_sim_j30/cos_sim_clean*100:.1f}%)")
print(f"   版权判定: {'✅ 通过' if cos_sim_j30 > 0.8 else '⚠️ 边缘' if cos_sim_j30 > 0.6 else '❌ 失败'}")
results.append(("JPEG Q=30", cos_sim_j30))

# 测试 5: 高斯模糊 σ=1.0
print("\n[测试 5] 高斯模糊 (Sigma = 1.0)")
blur_1_tensor = F_vision.gaussian_blur(T.ToTensor()(orig_img).unsqueeze(0), kernel_size=[5, 5], sigma=[1.0, 1.0])
blur_1 = T.ToPILImage()(blur_1_tensor.squeeze(0))
blur_1.save(os.path.join(output_dir, 'attack_blur_1.0.png'))
print(f"   已保存: attack_blur_1.0.png")
S_blur_1 = extract_signature(blur_1)
cos_sim_b1, _, _, _ = compare_signatures(S_orig, S_blur_1, mask_8x8)
print(f"   余弦相似度: {cos_sim_b1:.4f} (保留率: {cos_sim_b1/cos_sim_clean*100:.1f}%)")
print(f"   版权判定: {'✅ 通过' if cos_sim_b1 > 0.8 else '❌ 失败'}")
results.append(("高斯模糊 σ=1.0", cos_sim_b1))

# 测试 6: 高斯模糊 σ=2.0
print("\n[测试 6] 高斯模糊 (Sigma = 2.0)")
blur_2_tensor = F_vision.gaussian_blur(T.ToTensor()(orig_img).unsqueeze(0), kernel_size=[9, 9], sigma=[2.0, 2.0])
blur_2 = T.ToPILImage()(blur_2_tensor.squeeze(0))
blur_2.save(os.path.join(output_dir, 'attack_blur_2.0.png'))
print(f"   已保存: attack_blur_2.0.png")
S_blur_2 = extract_signature(blur_2)
cos_sim_b2, _, _, _ = compare_signatures(S_orig, S_blur_2, mask_8x8)
print(f"   余弦相似度: {cos_sim_b2:.4f} (保留率: {cos_sim_b2/cos_sim_clean*100:.1f}%)")
print(f"   版权判定: {'✅ 通过' if cos_sim_b2 > 0.8 else '⚠️ 边缘' if cos_sim_b2 > 0.6 else '❌ 失败'}")
results.append(("高斯模糊 σ=2.0", cos_sim_b2))

# 测试 7: 高斯模糊 σ=3.0
print("\n[测试 7] 高斯模糊 (Sigma = 3.0)")
blur_3_tensor = F_vision.gaussian_blur(T.ToTensor()(orig_img).unsqueeze(0), kernel_size=[11, 11], sigma=[3.0, 3.0])
blur_3 = T.ToPILImage()(blur_3_tensor.squeeze(0))
blur_3.save(os.path.join(output_dir, 'attack_blur_3.0.png'))
print(f"   已保存: attack_blur_3.0.png")
S_blur_3 = extract_signature(blur_3)
cos_sim_b3, _, _, _ = compare_signatures(S_orig, S_blur_3, mask_8x8)
print(f"   余弦相似度: {cos_sim_b3:.4f} (保留率: {cos_sim_b3/cos_sim_clean*100:.1f}%)")
print(f"   版权判定: {'✅ 通过' if cos_sim_b3 > 0.8 else '⚠️ 边缘' if cos_sim_b3 > 0.6 else '❌ 失败'}")
results.append(("高斯模糊 σ=3.0", cos_sim_b3))

# 测试 8: 中心裁剪 (模拟抠图攻击)
print("\n[测试 8] 中心裁剪 50% (模拟主体被移除)")
w, h = orig_img.size
left = w // 4
top = h // 4
right = 3 * w // 4
bottom = 3 * h // 4
cropped = orig_img.crop((left, top, right, bottom))
cropped = cropped.resize((512, 512), Image.Resampling.LANCZOS)
cropped.save(os.path.join(output_dir, 'attack_cropped.png'))
print(f"   已保存: attack_cropped.png")
S_cropped = extract_signature(cropped)
cos_sim_crop, _, _, _ = compare_signatures(S_orig, S_cropped, mask_8x8)
print(f"   余弦相似度: {cos_sim_crop:.4f}")
print(f"   说明: 主体被移除，分数应该显著下降")
results.append(("中心裁剪", cos_sim_crop))

# ==========================================
# 7. 结果汇总
# ==========================================
print("\n" + "="*60)
print("📊 [结果汇总表]")
print("="*60)
print(f"{'攻击类型':<20} {'相似度':>10} {'保留率':>10} {'状态':>8}")
print("-"*60)

baseline = results[0][1]
for name, score in results:
    retention = score / baseline * 100 if baseline > 0 else 0
    if score > 0.8:
        status = "✅ 强"
    elif score > 0.6:
        status = "⚠️ 中"
    elif score > 0.3:
        status = "🔶 弱"
    else:
        status = "❌ 无"
    print(f"{name:<20} {score:>10.4f} {retention:>9.1f}% {status:>8}")

print("="*60)
print("\n🎉 测试完成！")
print("\n[判定阈值说明]")
print("   > 0.80: 水印强保留，可可靠检测")
print("   0.60-0.80: 水印中度保留，可检测")
print("   0.30-0.60: 水印弱保留，边缘检测")
print("   < 0.30: 水印丢失，无法检测")
