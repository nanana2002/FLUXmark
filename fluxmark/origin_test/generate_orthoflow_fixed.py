import torch
from diffusers import FluxPipeline
import torchvision.transforms as T
import numpy as np
import os
import gc

# 设定缓存目录
os.environ['HF_HOME'] = '/data/daiyina/hf_cache'
output_dir = '/data/daiyina/project_flux/pic'
os.makedirs(output_dir, exist_ok=True)

# =====================================================================
# 🚀 阶段 1：极致优化加载模型
# =====================================================================
print(">>> [Phase 1] 正在加载 FLUX 流匹配模型...")
pipe = FluxPipeline.from_pretrained(
    '/data/daiyina/project_flux/model/flux-schnell',
    torch_dtype=torch.bfloat16
)

prompt = "a cat holding a sign that says hello world"

# 提取特征
print(">>> 正在编码 Prompt...")
with torch.no_grad():
    prompt_embeds, pooled_prompt_embeds, text_ids = pipe.encode_prompt(
        prompt=prompt, prompt_2=None, max_sequence_length=256
    )

# 移动到 CUDA
prompt_embeds = prompt_embeds.to("cuda")
pooled_prompt_embeds = pooled_prompt_embeds.to("cuda")
text_ids = text_ids.to("cuda")

# 卸载文本编码器节省显存
del pipe.text_encoder, pipe.text_encoder_2, pipe.tokenizer, pipe.tokenizer_2
pipe.text_encoder = None
pipe.text_encoder_2 = None
pipe.tokenizer = None
pipe.tokenizer_2 = None
gc.collect()
torch.cuda.empty_cache()
pipe.to("cuda")

# =====================================================================
# 🌊 阶段 2：密码学频域约束 (FFT Band-pass Generation)
# =====================================================================
print(">>> [Phase 2] 正在生成 FFT 中频密码本 W ...")
secret_key = 42
generator = torch.Generator(device='cuda').manual_seed(secret_key)
W_raw = torch.randn((1, 1024, 64), generator=generator, device='cuda', dtype=torch.float32)
W_spatial = W_raw.view(1, 32, 32, 64)

# 2D 傅里叶变换
F_W = torch.fft.fftshift(torch.fft.fft2(W_spatial, dim=(1, 2)), dim=(1, 2))

# 构造环形带通滤波器
h, w = 32, 32
Y, X = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
center_y, center_x = h // 2, w // 2
radius = torch.sqrt((Y - center_y)**2 + (X - center_x)**2).to('cuda')
mask_fft = ((radius >= 4) & (radius <= 12)).float().unsqueeze(0).unsqueeze(-1)

# 应用频域掩码并逆变换
F_W_filtered = F_W * mask_fft
W_filtered = torch.fft.ifft2(torch.fft.ifftshift(F_W_filtered, dim=(1, 2)), dim=(1, 2)).real
W = W_filtered.view(1, 1024, 64).to(torch.bfloat16)

# Token 级能量均衡
W = W / (torch.norm(W, dim=-1, keepdim=True) + 1e-8)

# =====================================================================
# 🧬 阶段 3：主体注意力掩码 (Semantic Anchoring)
# =====================================================================
print(">>> [Phase 3] 正在应用语义锚定...")
M_spatial = torch.zeros((1, 32, 32, 1), device='cuda', dtype=torch.bfloat16)
M_spatial[0, 6:26, 6:26, 0] = 1.0
M_attn = M_spatial.view(1, 1024, 1)

W_anchored = W * M_attn

# =====================================================================
# 📐 阶段 4：流形正交漂移注入
# =====================================================================
print(">>> [Phase 4] 开始正交流形生图 (4-Step ODE)...")
alpha = 0.35  # 略微提高注入强度

def orthogonal_drift_callback(pipe, step_index, timestep, callback_kwargs):
    latents = callback_kwargs["latents"]
    
    # Token-wise Gram-Schmidt 正交化
    dot_W_x = torch.sum(W_anchored * latents, dim=-1, keepdim=True)
    dot_x_x = torch.sum(latents * latents, dim=-1, keepdim=True) + 1e-8
    
    W_perp = W_anchored - (dot_W_x / dot_x_x) * latents
    
    # Token 级能量重归一化
    norm_W_anchored = torch.norm(W_anchored, dim=-1, keepdim=True)
    norm_W_perp = torch.norm(W_perp, dim=-1, keepdim=True) + 1e-8
    W_perp = W_perp * (norm_W_anchored / norm_W_perp)
    
    # 欧拉积分
    sigmas = pipe.scheduler.sigmas
    dt = sigmas[step_index + 1] - sigmas[step_index]
    latents = latents + alpha * W_perp * abs(dt)
    
    callback_kwargs["latents"] = latents
    return callback_kwargs

image_out = pipe(
    prompt_embeds=prompt_embeds,
    pooled_prompt_embeds=pooled_prompt_embeds,
    num_inference_steps=4,
    guidance_scale=0.0,
    height=512, width=512,
    callback_on_step_end=orthogonal_drift_callback,
).images[0]

img_path = os.path.join(output_dir, 'orthoflow_final_cat_fixed.png')
image_out.save(img_path)
print(f"✅ 水印图像已保存至: {img_path}")

# =====================================================================
# 🔐 阶段 5：差分签名提取
# =====================================================================
print(">>> [Phase 5] 正在提取并保存差分签名...")

img_tensor = T.ToTensor()(image_out).unsqueeze(0).to("cuda", dtype=torch.bfloat16)
img_tensor = (img_tensor - 0.5) * 2.0

with torch.no_grad():
    z_0 = pipe.vae.encode(img_tensor).latent_dist.sample()
    z_0 = (z_0 - pipe.vae.config.shift_factor) * pipe.vae.config.scaling_factor
    z_0 = pipe._pack_latents(z_0, batch_size=1, num_channels_latents=16, height=64, width=64)

    # 重构位置编码
    h = w = 32
    img_ids = torch.zeros(h, w, 3, device='cuda', dtype=torch.bfloat16)
    img_ids[..., 1] = torch.arange(h, device='cuda', dtype=torch.bfloat16).unsqueeze(1) / max(h - 1, 1)
    img_ids[..., 2] = torch.arange(w, device='cuda', dtype=torch.bfloat16).unsqueeze(0) / max(w - 1, 1)
    img_ids = img_ids.view(1, h * w, 3)

    # 使用相同的条件进行速度预测！
    t_detect = torch.tensor([pipe.scheduler.sigmas[-1]], device="cuda", dtype=torch.bfloat16)
    v_pred = pipe.transformer(
        hidden_states=z_0, timestep=t_detect / 1000,
        pooled_projections=pooled_prompt_embeds,  # 使用相同的 pooled_prompt_embeds
        encoder_hidden_states=prompt_embeds,      # 使用相同的 prompt_embeds
        txt_ids=text_ids,                         # 使用相同的 text_ids
        img_ids=img_ids, 
        return_dict=False,
    )[0]

# 计算 8x8 的余弦相似度矩阵
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
signature_path = os.path.join(output_dir, 'signature_orig_fixed.npy')
np.save(signature_path, S_orig)

# 打印统计信息
mask_8x8 = np.zeros((8, 8))
mask_8x8[1:7, 1:7] = 1.0
core_scores = S_orig[mask_8x8 == 1.0]
print(f"🎯 核心区域签名统计:")
print(f"   均值: {np.mean(core_scores):.4f}")
print(f"   标准差: {np.std(core_scores):.4f}")
print(f"   最小值: {np.min(core_scores):.4f}")
print(f"   最大值: {np.max(core_scores):.4f}")
print(f"✅ 签名已保存至: {signature_path}")

# 同时保存配置供检测使用
config = {
    'prompt': prompt,
    'secret_key': secret_key,
    'alpha': alpha,
    'mask_region': [6, 26, 6, 26],  # [y_start, y_end, x_start, x_end]
}
import json
with open(os.path.join(output_dir, 'watermark_config.json'), 'w') as f:
    json.dump(config, f)
print("🎉 生成完成！")
