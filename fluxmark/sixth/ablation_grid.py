#!/usr/bin/env python3
"""
消融实验：网格分辨率 (Grid Size) 对签名稳定性和定位精度的影响
对比 32×32、16×16、8×8 三种提取粒度


优化策略：按 grid_size 分段加载模型，每次只保留 VAE+Transformer 在 GPU，
避免 T5-XXL text encoder 与 Transformer 同时占用显存导致 OOM。
"""


import os
import json


with open("config.json", "r") as f:
    config = json.load(f)
os.environ["CUDA_VISIBLE_DEVICES"] = str(config.get("gpu_id", 0))
os.environ["HF_HOME"] = config["hf_cache"]


import torch
from diffusers import FluxPipeline
import torchvision.transforms as T
from PIL import Image
import numpy as np
from datetime import datetime
from tqdm import tqdm


output_dir = os.path.join(config["output_base_dir"], "result", "ablation_grid")
os.makedirs(output_dir, exist_ok=True)


watermarked_dir = os.path.join(config["output_base_dir"], "pic", "watermarked_img")
watermark_summary_path = os.path.join(
    watermarked_dir, f"{config['experiment_name']}_watermark_summary.json"
)
with open(watermark_summary_path, "r") as f:
    watermark_summary = json.load(f)
images_info = watermark_summary["images"]


print(f"📋 加载了 {len(images_info)} 张水印图像")
num_samples = min(len(images_info), 50)


grid_results = {
    32: {"means": [], "stds": [], "signatures": {}},
    16: {"means": [], "stds": [], "signatures": {}},
    8:  {"means": [], "stds": [], "signatures": {}},
}




def build_password_book():
    """重建密码本 W"""
    secret_key = config["secret_key"]
    g = torch.Generator(device="cuda").manual_seed(secret_key)
    W_raw = torch.randn((1, 1024, 64), generator=g, device="cuda", dtype=torch.float32)
    W_spatial = W_raw.view(1, 32, 32, 64)
    F_W = torch.fft.fftshift(torch.fft.fft2(W_spatial, dim=(1, 2)), dim=(1, 2))
    h, w = 32, 32
    Y, X = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    center_y, center_x = h // 2, w // 2
    radius = torch.sqrt((Y - center_y) ** 2 + (X - center_x) ** 2).to("cuda")
    r_inner = config["fft_radius_inner"]
    r_outer = config["fft_radius_outer"]
    mask_fft = (
        ((radius >= r_inner) & (radius <= r_outer)).float().unsqueeze(0).unsqueeze(-1)
    )
    F_W_filtered = F_W * mask_fft
    W_filtered = torch.fft.ifft2(
        torch.fft.ifftshift(F_W_filtered, dim=(1, 2)), dim=(1, 2)
    ).real
    W = W_filtered.view(1, 1024, 64).to(torch.bfloat16)
    W = W / (torch.norm(W, dim=-1, keepdim=True) + 1e-8)
    return W




def encode_all_prompts(pipe, images_info):
    """手动把 text encoder 搬上 GPU 编码所有 prompts，返回 CPU 缓存。
    不调用 pipe.encode_prompt，避免 diffusers 内部 device 推断与手动 to('cuda') 冲突。"""
    device = "cuda"
    text_encoder = pipe.text_encoder.to(device)
    text_encoder_2 = pipe.text_encoder_2.to(device)
    torch.cuda.empty_cache()


    cache = {}
    for img_info in tqdm(images_info, desc="编码 prompts"):
        prompt = img_info["prompt"]
        if prompt in cache:
            continue


        # CLIP (text_encoder) -> pooled_prompt_embeds
        text_inputs = pipe.tokenizer(
            prompt,
            padding="max_length",
            max_length=pipe.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids.to(device)
        with torch.no_grad():
            clip_embeds = text_encoder(text_input_ids, output_hidden_states=False)[0]
            pooled_prompt_embeds = clip_embeds[0:1, -1]  # [1, hidden_size]


        # T5 (text_encoder_2) -> prompt_embeds
        text_inputs_2 = pipe.tokenizer_2(
            prompt,
            padding="max_length",
            max_length=256,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids_2 = text_inputs_2.input_ids.to(device)
        with torch.no_grad():
            prompt_embeds = text_encoder_2(text_input_ids_2, output_hidden_states=False)[0]


        # FLUX transformer 需要的 text_ids
        text_ids = torch.zeros(prompt_embeds.shape[1], 3, device=device, dtype=prompt_embeds.dtype)


        cache[prompt] = {
            "prompt_embeds": prompt_embeds.cpu(),
            "pooled_prompt_embeds": pooled_prompt_embeds.cpu(),
            "text_ids": text_ids.cpu(),
        }
        del text_input_ids, text_input_ids_2, clip_embeds, pooled_prompt_embeds, prompt_embeds, text_ids


    # 编码完成后彻底释放 text encoder
    del pipe.text_encoder, pipe.text_encoder_2
    pipe.text_encoder = None
    pipe.text_encoder_2 = None
    torch.cuda.empty_cache()
    return cache




def extract_signature_grid(pipe, W, img_tensor, prompt_embeds, pooled_prompt_embeds, text_ids, grid_size):
    """提取指定 grid_size 的签名"""
    patch_size = 32 // grid_size
    prompt_embeds = prompt_embeds.to("cuda")
    pooled_prompt_embeds = pooled_prompt_embeds.to("cuda")
    text_ids = text_ids.to("cuda")
    with torch.no_grad():
        z_0 = pipe.vae.encode(img_tensor).latent_dist.sample()
        z_0 = (z_0 - pipe.vae.config.shift_factor) * pipe.vae.config.scaling_factor
        z_0 = pipe._pack_latents(
            z_0, batch_size=1, num_channels_latents=16, height=64, width=64
        )


        img_ids = torch.zeros(32, 32, 3, device="cuda", dtype=torch.bfloat16)
        img_ids[..., 1] = (
            torch.arange(32, device="cuda", dtype=torch.bfloat16).unsqueeze(1) / 31.0
        )
        img_ids[..., 2] = (
            torch.arange(32, device="cuda", dtype=torch.bfloat16).unsqueeze(0) / 31.0
        )
        img_ids = img_ids.view(1, 1024, 3)


        v_pred = pipe.transformer(
            hidden_states=z_0,
            timestep=torch.tensor(
                [pipe.scheduler.sigmas[-1]], device="cuda", dtype=torch.bfloat16
            )
            / 1000,
            pooled_projections=pooled_prompt_embeds,
            encoder_hidden_states=prompt_embeds,
            txt_ids=text_ids,
            img_ids=img_ids,
            return_dict=False,
        )[0]


    v_pred_spatial = v_pred.view(32, 32, 64)
    W_s = W.view(32, 32, 64)
    S = np.zeros((grid_size, grid_size))
    for i in range(grid_size):
        for j in range(grid_size):
            wp = (
                W_s[
                    i * patch_size : (i + 1) * patch_size,
                    j * patch_size : (j + 1) * patch_size,
                    :,
                ]
                .flatten()
                .float()
            )
            vp = (
                v_pred_spatial[
                    i * patch_size : (i + 1) * patch_size,
                    j * patch_size : (j + 1) * patch_size,
                    :,
                ]
                .flatten()
                .float()
            )
            S[i, j] = torch.nn.functional.cosine_similarity(
                wp.unsqueeze(0), vp.unsqueeze(0)
            ).item()
    return S




# ============================================================
# 主循环：每个 grid_size 独立加载模型、编码、提取、卸载
# ============================================================
for grid_size in [32, 16, 8]:
    print(f"\n{'='*50}")
    print(f"🔬 开始处理 grid_size = {grid_size}x{grid_size}")
    print(f"{'='*50}")


    # 1. 加载完整 pipeline 到 CPU
    print("🚀 加载 FLUX 模型到 CPU...")
    pipe = FluxPipeline.from_pretrained(
        config["model_path"], torch_dtype=torch.bfloat16
    )
    pipe.enable_attention_slicing(slice_size="auto")
    pipe.to("cpu")


    # 2. 编码 prompts（只在 GPU 上保留 text encoder）
    encoded_prompts_cache = encode_all_prompts(pipe, images_info[:num_samples])


    # 3. 把 VAE + Transformer 搬上 GPU，开始签名提取
    print("🚀 将 VAE 和 Transformer 搬到 GPU...")
    pipe.vae.to("cuda")
    pipe.transformer.to("cuda")
    torch.cuda.empty_cache()


    W = build_password_book()


    for img_info in tqdm(images_info[:num_samples], desc=f"Grid {grid_size}x{grid_size}"):
        img_id = img_info["id"]
        img_path = os.path.join(watermarked_dir, img_info["image_file"])
        img = Image.open(img_path).convert("RGB").resize((512, 512), Image.LANCZOS)
        img_tensor = T.ToTensor()(img).unsqueeze(0).to("cuda", dtype=torch.bfloat16)
        img_tensor = (img_tensor - 0.5) * 2.0


        enc = encoded_prompts_cache[img_info["prompt"]]
        S = extract_signature_grid(
            pipe, W, img_tensor,
            enc["prompt_embeds"],
            enc["pooled_prompt_embeds"],
            enc["text_ids"],
            grid_size,
        )
        grid_results[grid_size]["means"].append(float(np.mean(S)))
        grid_results[grid_size]["stds"].append(float(np.std(S)))
        grid_results[grid_size]["signatures"][img_id] = S.tolist()


        del img_tensor
        torch.cuda.empty_cache()


    # 4. 当前 grid_size 完成后彻底卸载模型
    del pipe, W, encoded_prompts_cache
    torch.cuda.empty_cache()
    print(f"✅ grid_size = {grid_size}x{grid_size} 完成，显存已释放")




# ============================================================
# 保存最终结果
# ============================================================
summary = {
    "experiment_name": config["experiment_name"],
    "timestamp": datetime.now().isoformat(),
    "num_samples": num_samples,
    "statistics": {
        "32x32": {
            "mean": float(np.mean(grid_results[32]["means"])),
            "std": float(np.mean(grid_results[32]["stds"])),
            "vector_dim": 64,
        },
        "16x16": {
            "mean": float(np.mean(grid_results[16]["means"])),
            "std": float(np.mean(grid_results[16]["stds"])),
            "vector_dim": 256,
        },
        "8x8": {
            "mean": float(np.mean(grid_results[8]["means"])),
            "std": float(np.mean(grid_results[8]["stds"])),
            "vector_dim": 1024,
        },
    },
    "details": {
        "32": {
            "means": grid_results[32]["means"],
            "stds": grid_results[32]["stds"],
            "signatures": grid_results[32]["signatures"],
        },
        "16": {
            "means": grid_results[16]["means"],
            "stds": grid_results[16]["stds"],
            "signatures": grid_results[16]["signatures"],
        },
        "8": {
            "means": grid_results[8]["means"],
            "stds": grid_results[8]["stds"],
            "signatures": grid_results[8]["signatures"],
        },
    },
}


summary_path = os.path.join(output_dir, "grid_ablation_summary.json")
with open(summary_path, "w") as f:
    json.dump(summary, f, indent=2)


print(f"\n✅ 网格分辨率消融实验完成！")
print(f"📊 结果保存至: {summary_path}")






