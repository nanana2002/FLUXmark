#!/usr/bin/env python3
"""快速测试：只提取一个 1024x1024 的攻击，验证修复是否有效并计时"""

import os, json, time
with open('config.json', 'r') as f:
    config = json.load(f)

os.environ['CUDA_VISIBLE_DEVICES'] = str(config.get('gpu_id', 0))
os.environ['HF_HOME'] = config.get('hf_cache', '/home/daiyn/hf_cache')

import torch
import numpy as np
from PIL import Image
import torchvision.transforms as T
from diffusers import FluxPipeline
import gc

model_path = config.get('model_path', '/home/daiyn/project_flux/model/flux-schnell')
device = 'cuda:0'

BASELINE_DIR = "pic/baseline_gaussionshading_img"
KEY_W_PATH = os.path.join(BASELINE_DIR, "baseline_gaussionshading_key_W.npy")

key_W = torch.from_numpy(np.load(KEY_W_PATH)).to(device, dtype=torch.float32)

prompt = "Elderly gray haired man in a suit scowling into the camera."

print("🚀 加载 FLUX 模型...")
pipe = FluxPipeline.from_pretrained(model_path, torch_dtype=torch.bfloat16).to(device)
with torch.no_grad():
    prompt_embeds, pooled_prompt_embeds, text_ids = pipe.encode_prompt(
        prompt=prompt, prompt_2=None, max_sequence_length=256
    )
pipe.text_encoder = None
pipe.text_encoder_2 = None
pipe.tokenizer = None
pipe.tokenizer_2 = None
gc.collect()
torch.cuda.empty_cache()
print("✅ FLUX 模型已精简")

def pack_latents(latents):
    b, c, h, w = latents.shape
    latents = latents.view(b, c, h // 2, 2, w // 2, 2)
    latents = latents.permute(0, 2, 4, 1, 3, 5)
    return latents.reshape(b, (h // 2) * (w // 2), c * 4)

def unpack_latents(latents, h=64, w=64):
    b, _, c4 = latents.shape
    c = c4 // 4
    latents = latents.view(b, h // 2, w // 2, c, 2, 2)
    latents = latents.permute(0, 3, 1, 4, 2, 5)
    return latents.reshape(b, c, h, w)

def extract_gs_score(image, key_W):
    img_tensor = T.ToTensor()(image).unsqueeze(0).to(device, dtype=torch.bfloat16)
    img_tensor = (img_tensor - 0.5) * 2.0
    _, _, img_h, img_w = img_tensor.shape
    latent_h, latent_w = img_h // 8, img_w // 8
    with torch.no_grad():
        z_0 = pipe.vae.encode(img_tensor).latent_dist.sample()
        z_0 = (z_0 - pipe.vae.config.shift_factor) * pipe.vae.config.scaling_factor
        z_t_packed = pack_latents(z_0)
        b, seq_len, c4 = z_t_packed.shape
        grid_h = latent_h // 2
        grid_w = latent_w // 2
        img_ids = torch.zeros(grid_h, grid_w, 3, device=device, dtype=torch.bfloat16)
        img_ids[..., 1] = torch.arange(grid_h, device=device, dtype=torch.bfloat16).unsqueeze(1) / max(grid_h - 1.0, 1.0)
        img_ids[..., 2] = torch.arange(grid_w, device=device, dtype=torch.bfloat16).unsqueeze(0) / max(grid_w - 1.0, 1.0)
        img_ids = img_ids.view(seq_len, 3)
        txt_ids_2d = text_ids if text_ids.dim() == 2 else text_ids.squeeze(0)
        sigmas = pipe.scheduler.sigmas
        print(f"   sigmas 长度: {len(sigmas)}, 步数: {len(sigmas)-1}")
        t0 = time.time()
        for i in range(len(sigmas) - 1, 0, -1):
            t_curr = sigmas[i]
            t_next = sigmas[i-1]
            v_pred = pipe.transformer(
                hidden_states=z_t_packed, timestep=torch.tensor([t_curr], device=device, dtype=torch.bfloat16) / 1000.0,
                pooled_projections=pooled_prompt_embeds, encoder_hidden_states=prompt_embeds,
                txt_ids=txt_ids_2d, img_ids=img_ids, return_dict=False,
            )[0]
            dt = t_next - t_curr
            z_t_packed = z_t_packed + v_pred * dt
        t1 = time.time()
        print(f"   提取耗时: {t1-t0:.1f}s")
        z_1_hat = unpack_latents(z_t_packed, h=latent_h, w=latent_w).to(torch.float32)
    orig_key_flat = key_W.flatten()
    extracted_flat = z_1_hat.flatten()
    score = torch.nn.functional.cosine_similarity(orig_key_flat.unsqueeze(0), extracted_flat.unsqueeze(0)).item()
    return score

# 测试 1024x1024 的 sdedit_0.3
test_path = "pic/baseline_gaussionshading_img/attacks/sdedit_0.3.png"
print(f"\n🔬 测试提取: {test_path}")
img = Image.open(test_path).convert("RGB")
print(f"   图像尺寸: {img.size}")
score = extract_gs_score(img, key_W)
print(f"   ✅ 提取分数: {score:.4f}")
