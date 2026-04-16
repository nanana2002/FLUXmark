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
alpha = 0.4 # 漂移强度（注入强度，后续实验可以调整测试 0.1 ~ 1.0）

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

def velocity_drift_callback(pipe, step_index, timestep, callback_kwargs):
    latents = callback_kwargs["latents"]

    # 获取当前的步长 dt
    sigmas = pipe.scheduler.sigmas
    dt = sigmas[step_index + 1] - sigmas[step_index] # 注意：在 FLUX 中 dt 通常是负数（从 1 到 0）

    # 核心公式：将漂移量积分进图像！
    # 因为我们要顺着生成方向漂移，FLUX 的 timestep 是从大到小，我们保持符号逻辑对齐
    latents = latents + alpha * W * abs(dt)

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

print(f"\n🎯 [检测结果] 提取的水印余弦相似度为: {cos_sim:.4f}")
if cos_sim > 0.1:  # 通常多维空间随机向量的相似度接近 0，大于 0.1 已经是天文级别的置信度了
    print("✅ 结论：水印检测成功！此图片受 FlowMark 版权保护。")
else:
    print("❌ 结论：未检测到水印。")
