import torch
from diffusers import FluxPipeline
import os
import gc

os.environ['HF_HOME'] = '/data/daiyina/hf_cache'

print("正在加载模型...")
pipe = FluxPipeline.from_pretrained(
    '/data/daiyina/project_flux/model/flux-schnell',
    torch_dtype=torch.bfloat16
)

prompt = "a cat holding a sign that says hello world"

# ========== 调试点 1：encode_prompt 之前检查 ==========
# print(f"\n[调试1] encode_prompt 前 pipe.text_encoder 类型: {type(pipe.text_encoder)}")
# print(f"[调试1] pipe.text_encoder 是否为 None: {pipe.text_encoder is None}")

# print("正在提取文本特征...")
with torch.no_grad():
    prompt_embeds, pooled_prompt_embeds, text_ids = pipe.encode_prompt(
        prompt=prompt, prompt_2=None, max_sequence_length=256
    )

# ========== 调试点 2：encode_prompt 之后检查 ==========
# print(f"\n[调试2] encode_prompt 后 pipe.text_encoder 类型: {type(pipe.text_encoder)}")
# print(f"[调试2] pipe.text_encoder 是否为 None: {pipe.text_encoder is None}")
# if hasattr(pipe.text_encoder, 'dtype'):
#     print(f"[调试2] pipe.text_encoder.dtype: {pipe.text_encoder.dtype}")

print("正在卸载文本编码器...")
pipe.text_encoder = None
pipe.text_encoder_2 = None
pipe.tokenizer = None
pipe.tokenizer_2 = None
gc.collect()
torch.cuda.empty_cache()

# ========== 调试点 3：删除后检查 ==========
# print(f"\n[调试3] 删除后 pipe.text_encoder 是否存在: {hasattr(pipe, 'text_encoder')}")
# try:
    # print(f"[调试3] pipe.text_encoder 值: {pipe.text_encoder}")
    # print(f"[调试3] pipe.text_encoder 类型: {type(pipe.text_encoder)}")
# except AttributeError as e:
    # print(f"[调试3] 访问 pipe.text_encoder 报错: {e}")

pipe.to("cuda")
print("模型常驻 GPU 完毕！")


# ==========================================
# 步骤 2：生成水印密钥（Velocity-Drift W）
# ==========================================
# FLUX 512x512 生成的 Latent token 形状固定为 (1, 1024, 64)
latent_shape = (1, 1024, 64)
alpha = 0.15 # 漂移强度（注入强度，后续实验可以调整测试 0.1 ~ 1.0）

# 设定一个伪随机种子作为我们的"私钥"
secret_key = 42
generator = torch.Generator(device='cuda').manual_seed(secret_key)

# 生成我们的网格漂移水印 W (符合高斯分布)
W = torch.randn(latent_shape, generator=generator, device='cuda', dtype=torch.bfloat16)

# ==========================================
# 步骤 3：速度场漂移注入 (The Forward ODE)
# ==========================================
# 在数学上，修改速度 v_new = v + alpha * W
# 等价于在欧拉积分步的末尾，使得 x_new = x_old + (v + alpha * W) * dt
# diffusers 提供了 callback，让我们能极其优雅地在每一步 dt 结束后注入偏移！

# ==========================================
# 核心创新：流形正交漂移 (Manifold-Orthogonal Drift)
# ==========================================
def velocity_drift_callback(pipe, step_index, timestep, callback_kwargs):
    latents = callback_kwargs["latents"]
    
    # 拿到我们固定的全局密钥 W
    global W 
    
    # ----------------------------------------------------
    # 👑 魔法发生的地方：动态计算正交投影 (Gram-Schmidt)
    # 我们要让 W 剔除掉与当前潜变量 latents 平行的成分
    # ----------------------------------------------------
    # 1. 计算内积 <W, latents> 和 <latents, latents>
    dot_W_x = torch.sum(W * latents)
    dot_x_x = torch.sum(latents * latents) + 1e-8 # 防止除零
    
    # 2. 计算投影并相减，得到绝对正交的 W_perp
    W_perp = W - (dot_W_x / dot_x_x) * latents
    
    # 为了保证注入强度稳定，我们把 W_perp 重新归一化到 W 的模长
    W_perp = W_perp * (torch.norm(W) / (torch.norm(W_perp) + 1e-8))
    # ----------------------------------------------------

    sigmas = pipe.scheduler.sigmas
    dt = sigmas[step_index + 1] - sigmas[step_index] 
    
    # 3. 使用正交水印进行无损注入！
    # 因为 W_perp 垂直于生成流形，我们可以把 alpha 稍微调大 (比如 0.2 或 0.3) 
    # 而绝对不会破坏猫的画质！
    latents = latents + 0.25 * W_perp * abs(dt) 

    callback_kwargs["latents"] = latents
    return callback_kwargs
    
print("正在以 FlowMark 水印注入生成图像...")
# 注意：我们这里传入了已经算好的 prompt_embeds，不用再算一次了

# 确保 embeddings 在 GPU 上（在调用 pipe() 之前）
prompt_embeds = prompt_embeds.to("cuda")
pooled_prompt_embeds = pooled_prompt_embeds.to("cuda")
text_ids = text_ids.to("cuda")

image_out = pipe(
    prompt_embeds=prompt_embeds,
    pooled_prompt_embeds=pooled_prompt_embeds,
    num_inference_steps=4,
    guidance_scale=0.0,
    height=512,
    width=512,
    callback_on_step_end=velocity_drift_callback, # 挂载我们的水印注入函数！
).images[0]

output_path = '/data/daiyina/project_flux/pic/watermarked_cat.png'
image_out.save(output_path)
print(f"带水印的图片已保存到: {output_path}")

# ==========================================
# 步骤 4：免训练逆向提取 (The Reverse ODE Detection)
# ==========================================
print("\n--- 开始执行水印检测 ---")

# 1. 模拟现实场景：我们只拿到了一张图片（甚至可能被压缩过），我们把它重新转回潜空间 (Latent)
import torchvision.transforms as T
img_tensor = T.ToTensor()(image_out).unsqueeze(0).to("cuda", dtype=torch.bfloat16)
# 将像素归一化到 [-1, 1] (FLUX VAE 的标准要求)
img_tensor = (img_tensor - 0.5) * 2.0

with torch.no_grad():
    # 使用 VAE 编码图片得到 x_0
    z_0 = pipe.vae.encode(img_tensor).latent_dist.sample()
    z_0 = (z_0 - pipe.vae.config.shift_factor) * pipe.vae.config.scaling_factor
    # 把 [1, 16, 64, 64] 转换成 FLUX Transformer 需要的[1, 1024, 64] 格式
    z_0 = pipe._pack_latents(z_0, batch_size=1, num_channels_latents=16, height=64, width=64)

# 创建 img_ids 用于位置编码 (手动创建，替代 pipe._get_image_ids)
h = w = 32  # 64/2 = 32 patches
img_ids = torch.zeros(h, w, 3, device='cuda', dtype=torch.bfloat16)
img_ids[..., 1] = torch.arange(h, device='cuda', dtype=torch.bfloat16).unsqueeze(1) / max(h - 1, 1)
img_ids[..., 2] = torch.arange(w, device='cuda', dtype=torch.bfloat16).unsqueeze(0) / max(w - 1, 1)
img_ids = img_ids.view(1, h * w, 3)

# 2. 估计生成此图时的末端"速度残差"
# 因为图是顺着 W 的方向被强行偏移的，当我们把 z_0 送进模型，模型会觉得"这个图偏离了自然分布"
# 它输出的速度向量 v_pred 中，就会极其强烈地包含我们的 W 信号！
with torch.no_grad():
    # 设定 timestep 为接近 0 的状态（图已经生成完的状态）
    t_detect = torch.tensor([pipe.scheduler.sigmas[-1]], device="cuda", dtype=torch.bfloat16)

    # 让模型预测速度
    v_pred = pipe.transformer(
        hidden_states=z_0,
        timestep=t_detect / 1000, # diffusers 内部 scaling
        # guidance=torch.tensor([0.0], device="cuda", dtype=torch.bfloat16),
        pooled_projections=pooled_prompt_embeds,
        encoder_hidden_states=prompt_embeds,
        txt_ids=text_ids,
        img_ids=img_ids,  # 使用手动创建的 img_ids
        return_dict=False,
    )[0]

# 3. 计算相关性 (Cosine Similarity)
# 扁平化以便计算
W_flat = W.flatten().float()
v_pred_flat = v_pred.flatten().float()

# 计算余弦相似度
cos_sim = torch.nn.functional.cosine_similarity(W_flat.unsqueeze(0), v_pred_flat.unsqueeze(0)).item()

# ---------------- 此代码替换掉你脚本最后一步的相似度计算 ----------------
import matplotlib.pyplot as plt
import numpy as np

# W 的形状是[1, 1024, 64]
# v_pred 的形状是 [1, 1024, 64]
# 这 1024 个 token 实际上对应了 32x32 的空间网格 (32 * 32 = 1024)
grid_size = 32

# 1. 把 W 和 v_pred 还原回 32x32 的空间结构
W_spatial = W.view(grid_size, grid_size, 64)          #[32, 32, 64]
v_pred_spatial = v_pred.view(grid_size, grid_size, 64)# [32, 32, 64]

# 2. 我们使用 4x4 的滑动窗口（Patch）来计算局部相似度
# 这将把 32x32 的网格缩小成 8x8 的热力图
patch_size = 4
heatmap_size = grid_size // patch_size
heatmap = np.zeros((heatmap_size, heatmap_size))

for i in range(heatmap_size):
    for j in range(heatmap_size):
        # 提取局部的 W 和 v_pred (尺寸为 4 x 4 x 64)
        h_start, h_end = i * patch_size, (i + 1) * patch_size
        w_start, w_end = j * patch_size, (j + 1) * patch_size
        
        W_patch = W_spatial[h_start:h_end, w_start:w_end, :].flatten().float()
        v_patch = v_pred_spatial[h_start:h_end, w_start:w_end, :].flatten().float()
        
        # 计算局部的余弦相似度
        patch_sim = torch.nn.functional.cosine_similarity(W_patch.unsqueeze(0), v_patch.unsqueeze(0)).item()
        heatmap[i, j] = patch_sim

# 3. 打印全局相似度作为参考
global_sim = np.mean(heatmap)
print(f"\n🎯 [全局版权验证] 平均相似度为: {global_sim:.4f}")

# 4. 绘制并保存热力图 (这就是你要放到论文里的图！)
plt.figure(figsize=(6, 5))
# 使用 cmap='RdYlGn' (红-黄-绿)，绿色代表相似度高（未篡改），红色代表相似度低（被篡改）
plt.imshow(heatmap, cmap='RdYlGn', interpolation='nearest', vmin=0.0, vmax=0.5) 
plt.colorbar(label='Cosine Similarity')
plt.title('FlowMark Tamper Localization Heatmap')
plt.axis('off')

heatmap_path = '/data/daiyina/project_flux/pic/heatmap_original.png'
plt.savefig(heatmap_path, bbox_inches='tight', dpi=300)
print(f"✅ 热力图已保存至: {heatmap_path}")

print(f"\n🎯 [检测结果] 提取的水印余弦相似度为: {cos_sim:.4f}")
if cos_sim > 0.1:  # 通常多维空间随机向量的相似度接近 0，大于 0.1 已经是天文级别的置信度了
    print("✅ 结论：水印检测成功！此图片受 FlowMark 版权保护。")
else:
    print("❌ 结论：未检测到水印。")
