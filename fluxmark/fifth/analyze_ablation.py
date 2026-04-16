#!/usr/bin/env python3
"""
消融实验分析脚本
对比三种消融版本与完整版（+无水印基准）的差异：
1. 不加 FFT 约束 (ablation_fft)
2. 不加正交投影 (ablation_obj)  
3. 语义绑定掩码 (ablation_sem)


评估维度：
- 隐蔽性：FID（vs 无水印图）+ CLIP Score
- 水印存在性：核心区域签名均值（判断消融后是否还有水印信号）
"""


import os
import json


with open('config.json', 'r') as f:
    config = json.load(f)
os.environ['CUDA_VISIBLE_DEVICES'] = str(config.get('gpu_id', 0))
os.environ['HF_HOME'] = config['hf_cache']


import torch
import torchvision.transforms as T
from PIL import Image
import numpy as np
from datetime import datetime
from tqdm import tqdm
from scipy import linalg
from diffusers import FluxPipeline


try:
    from torchmetrics.multimodal.clip_score import CLIPScore
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False


try:
    from torchmetrics.image.fid import FrechetInceptionDistance
    FID_AVAILABLE = True
except ImportError:
    FID_AVAILABLE = False


result_dir = os.path.join(config['output_base_dir'], 'result')
os.makedirs(result_dir, exist_ok=True)


base_pic_dir = os.path.join(config['output_base_dir'], 'pic')
no_wm_dir = os.path.join(base_pic_dir, 'no_watermarked_img')
full_wm_dir = os.path.join(base_pic_dir, 'watermarked_img')


ABLATIONS = {
    'full': {
        'name': 'Full Method',
        'dir': full_wm_dir,
        'prefix': 'wm',
        'desc': '完整方法 (FFT + Orthogonal + Semantic Anchor)'
    },
    'no_fft': {
        'name': 'w/o FFT',
        'dir': os.path.join(base_pic_dir, 'ablation_fft_watermarked_img'),
        'prefix': 'ablation_fft',
        'desc': '消融：不加 FFT 约束'
    },
    'no_ortho': {
        'name': 'w/o Orthogonal',
        'dir': os.path.join(base_pic_dir, 'ablation_obj_watermarked_img'),
        'prefix': 'ablation_obj',
        'desc': '消融：不加正交投影'
    },
    'semantic': {
        'name': 'w/ Semantic Only',
        'dir': os.path.join(base_pic_dir, 'ablation_sem_watermarked_img'),
        'prefix': 'ablation_sem',
        'desc': '消融：仅用语义绑定掩码'
    }
}


print("="*60)
print("🔬 消融实验分析")
print("="*60)


# 加载无水印摘要
no_wm_summary_path = os.path.join(no_wm_dir, f'{config["experiment_name"]}_no_watermark_summary.json')
with open(no_wm_summary_path, 'r') as f:
    no_wm_summary = json.load(f)
num_images = len(no_wm_summary['images'])
print(f"基准样本数: {num_images} 张")


# 检查各消融目录是否存在
for key, info in ABLATIONS.items():
    exists = os.path.exists(info['dir']) and len(os.listdir(info['dir'])) > 0
    print(f"   {info['name']}: {'✅' if exists else '❌'} ({info['dir']})")


# ==========================================
# 辅助函数
# ==========================================
def load_images_batch(image_paths, size=(512, 512)):
    images = []
    for path in image_paths:
        img = Image.open(path).convert('RGB')
        if img.size != size:
            img = img.resize(size, Image.Resampling.LANCZOS)
        images.append(T.ToTensor()(img))
    return torch.stack(images)


def compute_fid(images_real, images_fake, device='cuda'):
    if not FID_AVAILABLE:
        return None
    images_real = (images_real * 255).byte()
    images_fake = (images_fake * 255).byte()
    fid = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
    batch_size = 50
    for i in range(0, len(images_real), batch_size):
        fid.update(images_real[i:i+batch_size].to(device), real=True)
        fid.update(images_fake[i:i+batch_size].to(device), real=False)
    return fid.compute().item()


def compute_clip_score(images, prompts, device='cuda'):
    if not CLIP_AVAILABLE:
        return None
    images_uint8 = (images * 255).byte()
    clip_fn = CLIPScore(model_name_or_path="openai/clip-vit-base-patch16").to(device)
    scores = []
    batch_size = 25
    for i in range(0, len(images_uint8), batch_size):
        scores.append(clip_fn(images_uint8[i:i+batch_size].to(device), prompts[i:i+batch_size]).item())
    return np.mean(scores)


def extract_signature_mean(img_path, pipe, prompt, W_anchored, mask_8x8):
    """快速提取单张图的 8x8 签名核心区域均值"""
    img = Image.open(img_path).convert('RGB').resize((512, 512), Image.Resampling.LANCZOS)
    img_tensor = T.ToTensor()(img).unsqueeze(0).to("cuda", dtype=torch.bfloat16)
    img_tensor = (img_tensor - 0.5) * 2.0


    with torch.no_grad():
        z_0 = pipe.vae.encode(img_tensor).latent_dist.sample()
        z_0 = (z_0 - pipe.vae.config.shift_factor) * pipe.vae.config.scaling_factor
        z_0 = pipe._pack_latents(z_0, batch_size=1, num_channels_latents=16, height=64, width=64)


        prompt_embeds, pooled_prompt_embeds, text_ids = pipe.encode_prompt(prompt=prompt, prompt_2=None, max_sequence_length=256)
        img_ids = torch.zeros(32, 32, 3, device='cuda', dtype=torch.bfloat16)
        img_ids[..., 1] = torch.arange(32, device='cuda', dtype=torch.bfloat16).unsqueeze(1) / 31.0
        img_ids[..., 2] = torch.arange(32, device='cuda', dtype=torch.bfloat16).unsqueeze(0) / 31.0
        img_ids = img_ids.view(1, 1024, 3)


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
    W_anchored_spatial = W_anchored.view(32, 32, 64)
    S = np.zeros((8, 8))
    for i in range(8):
        for j in range(8):
            W_patch = W_anchored_spatial[i*4:(i+1)*4, j*4:(j+1)*4, :].flatten().float()
            v_patch = v_pred_spatial[i*4:(i+1)*4, j*4:(j+1)*4, :].flatten().float()
            S[i, j] = torch.nn.functional.cosine_similarity(W_patch.unsqueeze(0), v_patch.unsqueeze(0)).item()


    return float(np.mean(S[mask_8x8 == 1.0]))


# ==========================================
# 1. 隐蔽性分析 (FID + CLIP)
# ==========================================
fid_results = {}
clip_results = {}


if FID_AVAILABLE or CLIP_AVAILABLE:
    print("\n" + "="*60)
    print("📊 隐蔽性分析")
    print("="*60)


    # 加载无水印图像
    no_wm_paths = [os.path.join(no_wm_dir, no_wm_summary['images'][i]['image_file']) for i in range(num_images)]
    no_wm_paths = [p for p in no_wm_paths if os.path.exists(p)]
    print(f"\n加载无水印图: {len(no_wm_paths)} 张")
    no_wm_images = load_images_batch(no_wm_paths)


    prompts = [no_wm_summary['images'][i]['prompt'] for i in range(len(no_wm_paths))]


    for key, info in ABLATIONS.items():
        ab_paths = [os.path.join(info['dir'], f"{info['prefix']}_{i:03d}.png") for i in range(len(no_wm_paths))]
        ab_paths = [p for p in ab_paths if os.path.exists(p)]
        if len(ab_paths) < len(no_wm_paths):
            print(f"   ⚠️ {info['name']}: 只找到 {len(ab_paths)}/{len(no_wm_paths)} 张")
        if not ab_paths:
            continue


        ab_images = load_images_batch(ab_paths)


        if FID_AVAILABLE:
            n = min(len(no_wm_images), len(ab_images))
            fid = compute_fid(no_wm_images[:n], ab_images[:n])
            fid_results[key] = fid
            print(f"   {info['name']:<20} FID = {fid:.2f}")


        if CLIP_AVAILABLE:
            n = min(len(ab_images), len(prompts))
            clip = compute_clip_score(ab_images[:n], prompts[:n])
            clip_results[key] = clip
            print(f"   {info['name']:<20} CLIP = {clip:.4f}")


# ==========================================
# 2. 水印签名提取（GPU 批量）
# ==========================================
print("\n" + "="*60)
print("🔐 水印签名提取（GPU 模式）")
print("="*60)


print("🚀 加载 FLUX 模型...")
pipe = FluxPipeline.from_pretrained(config['model_path'], torch_dtype=torch.bfloat16)
pipe.to("cuda")
# pipe.enable_vae_tiling()
pipe.enable_attention_slicing(slice_size="auto")


secret_key = config['secret_key']
g = torch.Generator(device='cuda').manual_seed(secret_key)
W_raw = torch.randn((1, 1024, 64), generator=g, device='cuda', dtype=torch.float32)
W_spatial = W_raw.view(1, 32, 32, 64)
F_W = torch.fft.fftshift(torch.fft.fft2(W_spatial, dim=(1, 2)), dim=(1, 2))
h, w = 32, 32
Y, X = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
center_y, center_x = h // 2, w // 2
radius = torch.sqrt((Y - center_y)**2 + (X - center_x)**2).to('cuda')
r_inner = config['fft_radius_inner']
r_outer = config['fft_radius_outer']
mask_fft = ((radius >= r_inner) & (radius <= r_outer)).float().unsqueeze(0).unsqueeze(-1)
F_W_filtered = F_W * mask_fft
W_filtered = torch.fft.ifft2(torch.fft.ifftshift(F_W_filtered, dim=(1, 2)), dim=(1, 2)).real
W = W_filtered.view(1, 1024, 64).to(torch.bfloat16)
W = W / (torch.norm(W, dim=-1, keepdim=True) + 1e-8)
M_spatial = torch.zeros((1, 32, 32, 1), device='cuda', dtype=torch.bfloat16)
y_s, y_e, x_s, x_e = config['mask_region']
M_spatial[0, y_s:y_e, x_s:x_e, 0] = 1.0
M = M_spatial.view(1, 1024, 1)
W_anchored = W * M


mask_8x8 = np.zeros((8, 8))
mask_8x8[1:7, 1:7] = 1.0


signature_means = {}
for key, info in ABLATIONS.items():
    ab_dir = info['dir']
    scores = []
    for i in tqdm(range(min(num_images, 50)), desc=f"{info['name']}", leave=False):  # 先抽50张快速看
        prefix = info['prefix']
        img_path = os.path.join(ab_dir, f"{prefix}_{i:03d}.png")
        if not os.path.exists(img_path):
            continue
        prompt = no_wm_summary['images'][i]['prompt']
        try:
            s = extract_signature_mean(img_path, pipe, prompt, W_anchored, mask_8x8)
            scores.append(s)
        except Exception as e:
            pass
    if scores:
        signature_means[key] = {
            'mean': float(np.mean(scores)),
            'std': float(np.std(scores)),
            'num_samples': len(scores)
        }
        print(f"   {info['name']:<20} S_mean = {np.mean(scores):.4f} ± {np.std(scores):.4f} (n={len(scores)})")


# ==========================================
# 3. 保存结果
# ==========================================
output = {
    'experiment_name': config['experiment_name'],
    'timestamp': datetime.now().isoformat(),
    'fid': fid_results,
    'clip_score': clip_results,
    'signature_mean': signature_means
}


output_path = os.path.join(result_dir, 'ablation_analysis.json')
with open(output_path, 'w') as f:
    json.dump(output, f, indent=2)


print(f"\n✅ 结果已保存: {output_path}")


# ==========================================
# 4. 打印 LaTeX 表格
# ==========================================
print("\n" + "="*60)
print("📄 LaTeX 表格格式")
print("="*60)


print("\n% Table: Ablation Study (Invisibility + Watermark Strength)")
print("\\begin{table}[t]")
print("\\centering")
print("\\caption{Ablation Study Results}")
print("\\label{tab:ablation}")
print("\\resizebox{0.98\\columnwidth}{!}{%")
print("\\begin{tabular}{lccc}")
print("\\toprule")
print("\\textbf{Method} & \\textbf{FID$\\downarrow$} & \\textbf{CLIP Score$\\uparrow$} & \\textbf{S$_{mean}$} \\\\")
print("\\midrule")


order = ['full', 'no_fft', 'no_ortho', 'semantic']
for key in order:
    if key not in ABLATIONS:
        continue
    name = ABLATIONS[key]['name']
    fid = fid_results.get(key)
    clip = clip_results.get(key)
    sig = signature_means.get(key)
    fid_str = f"{fid:.2f}" if fid is not None else "-"
    clip_str = f"{clip:.2f}" if clip is not None else "-"
    sig_str = f"{sig['mean']:.4f}" if sig else "-"
    print(f"{name} & {fid_str} & {clip_str} & {sig_str} \\\\")


print("\\bottomrule")
print("\\end{tabular}%")
print("}")
print("\\end{table}")






