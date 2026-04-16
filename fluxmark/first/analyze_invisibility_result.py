#!/usr/bin/env python3
"""
隐蔽性分析脚本（顶会标准版）
使用 FID 和 CLIP Score 评估水印对图像质量的影响
替代传统的 PSNR/SSIM（像素级指标不适合扩散模型水印）

FID (Fréchet Inception Distance): 衡量分布真实性，越低越好
CLIP Score: 衡量图文语义一致性，越高越好
"""

# ===================== 【修复 1：国内镜像加速】=====================
import os
# 强制使用 Hugging Face 国内镜像，解决下载慢/卡住问题
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# ================================================================

import torch
import torchvision.transforms as T
from PIL import Image
import numpy as np
import json
from datetime import datetime
from tqdm import tqdm
from scipy import linalg

# 加载配置
with open('config.json', 'r') as f:
    config = json.load(f)

os.environ['HF_HOME'] = config['hf_cache']

# 路径设置
result_dir = os.path.join(config['output_base_dir'], 'result')
os.makedirs(result_dir, exist_ok=True)

no_watermark_dir = os.path.join(config['output_base_dir'], 'pic', 'no_watermarked_img')
watermarked_dir = os.path.join(config['output_base_dir'], 'pic', 'watermarked_img')

# ==========================================
# 1. 尝试加载 CLIP Score
# ==========================================
try:
    from torchmetrics.multimodal.clip_score import CLIPScore
    CLIP_AVAILABLE = True
    print("✅ CLIP Score 评估可用")
except ImportError:
    CLIP_AVAILABLE = False
    print("⚠️  CLIP Score 未安装，跳过语义一致性评估")
    print("   安装命令: pip install torchmetrics")

# ==========================================
# 2. 尝试加载 FID
# ==========================================
try:
    from torchmetrics.image.fid import FrechetInceptionDistance
    FID_AVAILABLE = True
    print("✅ FID 评估可用")
except ImportError:
    FID_AVAILABLE = False
    print("⚠️  FID 未安装，跳过分布局实性评估")
    print("   安装命令: pip install torchmetrics[image]")

# ==========================================
# 3. 辅助函数
# ==========================================

def load_images_batch(image_paths, size=(512, 512)):
    """批量加载图像为Tensor"""
    images = []
    for path in image_paths:
        img = Image.open(path).convert('RGB')
        if img.size != size:
            img = img.resize(size, Image.Resampling.LANCZOS)
        img_tensor = T.ToTensor()(img)  # [0, 1]
        images.append(img_tensor)
    return torch.stack(images)


def compute_fid(images_real, images_fake, device='cuda'):
    """
    计算FID (Fréchet Inception Distance)
    
    Args:
        images_real: 真实图像批次 [N, 3, H, W], 值域 [0, 1]
        images_fake: 生成图像批次 [N, 3, H, W], 值域 [0, 1]
    
    Returns:
        fid_score: FID分数，越低越好
    """
    if not FID_AVAILABLE:
        return None
    
    # 转换到 [0, 255] uint8
    images_real = (images_real * 255).byte()
    images_fake = (images_fake * 255).byte()
    
    # 初始化FID模型
    fid = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
    
    # 分批计算（避免OOM）
    batch_size = 20
    for i in range(0, len(images_real), batch_size):
        real_batch = images_real[i:i+batch_size].to(device)
        fake_batch = images_fake[i:i+batch_size].to(device)
        fid.update(real_batch, real=True)
        fid.update(fake_batch, real=False)
    
    fid_score = fid.compute().item()
    return fid_score


def compute_clip_score(images, prompts, device='cuda'):
    """
    本地 CLIP 计算，100% 稳定不报错
    """
    import torch
    from transformers import CLIPProcessor, CLIPModel

    # ============ 直接用本地模型 ============
    LOCAL_CLIP_PATH = "/data/daiyina/project_flux/model/clip-model"   # <-- 就是你刚才保存的路径
    # =======================================

    model = CLIPModel.from_pretrained(LOCAL_CLIP_PATH).to(device)
    processor = CLIPProcessor.from_pretrained(LOCAL_CLIP_PATH)

    images_np = (images * 255).byte().cpu().numpy()
    images_np = images_np.transpose(0, 2, 3, 1)  # [B, H, W, C]

    scores = []
    batch_size = 4

    for i in range(0, len(images_np), batch_size):
        img_batch = images_np[i:i+batch_size]
        prompt_batch = prompts[i:i+batch_size]

        # 正确预处理
        inputs = processor(
            text=prompt_batch,
            images=img_batch,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        score = outputs.logits_per_image.diag().cpu().numpy()
        scores.extend(score)

    return float(np.mean(scores))

# ==========================================
# 4. 加载图像信息
# ==========================================
print("📋 加载图像信息...")

no_wm_summary_path = os.path.join(no_watermark_dir, f'{config["experiment_name"]}_no_watermark_summary.json')
watermark_summary_path = os.path.join(watermarked_dir, f'{config["experiment_name"]}_watermark_summary.json')

if not os.path.exists(no_wm_summary_path):
    print(f"错误: 未找到无水印图像摘要: {no_wm_summary_path}")
    exit(1)

if not os.path.exists(watermark_summary_path):
    print(f"错误: 未找到水印图像摘要: {watermark_summary_path}")
    exit(1)

with open(no_wm_summary_path, 'r') as f:
    no_wm_summary = json.load(f)

with open(watermark_summary_path, 'r') as f:
    watermark_summary = json.load(f)

print(f"   无水印图像: {len(no_wm_summary['images'])}")
print(f"   水印图像: {len(watermark_summary['images'])}")

# ==========================================
# 5. 计算 FID
# ==========================================
fid_score = None
if FID_AVAILABLE:
    print("\n🎯 计算 FID (Fréchet Inception Distance)...")
    print("   说明: FID 衡量两组图像的分布差异，越低越好 (<50优秀, <100良好)")
    
    num_samples = min(len(no_wm_summary['images']), len(watermark_summary['images']), 100)
    
    # 加载图像路径
    real_paths = []
    fake_paths = []
    
    for i in range(num_samples):
        no_wm_file = no_wm_summary['images'][i]['image_file']
        wm_file = watermark_summary['images'][i]['image_file']
        
        real_path = os.path.join(no_watermark_dir, no_wm_file)
        fake_path = os.path.join(watermarked_dir, wm_file)
        
        if os.path.exists(real_path) and os.path.exists(fake_path):
            real_paths.append(real_path)
            fake_paths.append(fake_path)
    
    print(f"   使用 {len(real_paths)} 对图像计算 FID...")
    
    # 加载图像
    print("   加载无水印图像...")
    images_real = load_images_batch(real_paths).cpu()
    
    print("   加载水印图像...")
    images_fake = load_images_batch(fake_paths).cpu()
    
    print("   计算 FID (可能需要几分钟)...")
    try:
        fid_score = compute_fid(images_real, images_fake)
        print(f"   ✅ FID 计算完成: {fid_score:.2f}")
    except Exception as e:
        print(f"   ❌ FID 计算失败: {e}")
        fid_score = None
else:
    print("\n⏭️  跳过 FID 计算（未安装）")

# ==========================================
# 6. 计算 CLIP Score
# ==========================================
clip_score_no_wm = None
clip_score_wm = None

if CLIP_AVAILABLE:
    print("\n🎯 计算 CLIP Score (图文语义一致性)...")
    print("   说明: CLIP Score 衡量图像与Prompt的匹配度，越高越好")
    
    num_samples = min(len(no_wm_summary['images']), len(watermark_summary['images']), 50)
    
    # 准备数据
    no_wm_images = []
    wm_images = []
    prompts = []
    
    for i in range(num_samples):
        no_wm_info = no_wm_summary['images'][i]
        wm_info = watermark_summary['images'][i]
        
        no_wm_path = os.path.join(no_watermark_dir, no_wm_info['image_file'])
        wm_path = os.path.join(watermarked_dir, wm_info['image_file'])
        prompt = no_wm_info['prompt']
        
        if os.path.exists(no_wm_path) and os.path.exists(wm_path):
            no_wm_img = T.ToTensor()(Image.open(no_wm_path).convert('RGB'))
            wm_img = T.ToTensor()(Image.open(wm_path).convert('RGB'))
            
            no_wm_images.append(no_wm_img)
            wm_images.append(wm_img)
            prompts.append(prompt)
    
    print(f"   使用 {len(prompts)} 张图像计算 CLIP Score...")
    
    # 堆叠为批次
    no_wm_batch = torch.stack(no_wm_images)
    wm_batch = torch.stack(wm_images)
    
    try:
        print("   计算无水印图像 CLIP Score...")
        clip_score_no_wm = compute_clip_score(no_wm_batch, prompts)
        
        print("   计算水印图像 CLIP Score...")
        clip_score_wm = compute_clip_score(wm_batch, prompts)
        
        print(f"   ✅ CLIP Score 计算完成")
    except Exception as e:
        print(f"   ❌ CLIP Score 计算失败: {e}")
        clip_score_no_wm = None
        clip_score_wm = None
else:
    print("\n⏭️  跳过 CLIP Score 计算（未安装）")

# ==========================================
# 7. 保存结果
# ==========================================
output = {
    'experiment_name': config['experiment_name'],
    'timestamp': datetime.now().isoformat(),
    'num_samples_fid': len(real_paths) if fid_score else 0,
    'num_samples_clip': len(prompts) if clip_score_wm else 0,
    'fid': {
        'score': fid_score,
        'interpretation': 'lower is better, <50 excellent, <100 good, <200 moderate'
    },
    'clip_score': {
        'no_watermark': clip_score_no_wm,
        'watermarked': clip_score_wm,
        'difference': (clip_score_wm - clip_score_no_wm) if clip_score_wm and clip_score_no_wm else None,
        'interpretation': 'higher is better, should be very close between two groups'
    }
}

output_path = os.path.join(result_dir, 'invisibility_analysis.json')
with open(output_path, 'w') as f:
    json.dump(output, f, indent=2)

print(f"\n✅ 结果已保存至: {output_path}")

# ==========================================
# 8. 打印汇总
# ==========================================
print("\n" + "="*80)
print("📊 隐蔽性分析结果 (顶会标准评估)")
print("="*80)

if fid_score:
    print(f"\n🎯 FID (Fréchet Inception Distance)")
    print(f"   分数: {fid_score:.2f}")
    if fid_score < 50:
        print(f"   评价: 🟢 优秀 (分布几乎无差异)")
    elif fid_score < 100:
        print(f"   评价: 🟡 良好 (轻微差异)")
    elif fid_score < 200:
        print(f"   评价: 🟠 中等 (有一定差异)")
    else:
        print(f"   评价: 🔴 较差 (分布差异明显)")
    print(f"   说明: 衡量两组图像在特征空间的分布距离")

if clip_score_wm and clip_score_no_wm:
    print(f"\n🎯 CLIP Score (图文语义一致性)")
    print(f"   无水印图像: {clip_score_no_wm:.4f}")
    print(f"   水印图像:   {clip_score_wm:.4f}")
    diff = clip_score_wm - clip_score_no_wm
    print(f"   差异:       {diff:+.4f} ({diff/abs(clip_score_no_wm)*100:+.2f}%)")
    
    if abs(diff) < 0.5:
        print(f"   评价: 🟢 语义一致性极佳")
    elif abs(diff) < 1.0:
        print(f"   评价: 🟡 语义一致性良好")
    else:
        print(f"   评价: 🟠 语义有一定偏移")
    
    print(f"   说明: 衡量图像与Prompt的语义匹配度")
    print(f"   结论: 如果两个分数极其接近，证明水印【零语义干扰】！")

print("\n" + "="*80)
print("\n📄 LaTeX表格格式:")
print("-"*80)
print("\\begin{table}[h]")
print("\\centering")
print("\\caption{Invisibility Evaluation (FID and CLIP Score)}")
print("\\begin{tabular}{lcc}")
print("\\hline")
print("Metric & Value & Interpretation \\\\")
print("\\hline")
if fid_score:
    print(f"FID ($\\downarrow$) & {fid_score:.2f} & " + ("Excellent" if fid_score < 50 else "Good" if fid_score < 100 else "Moderate") + " \\\\")
if clip_score_wm and clip_score_no_wm:
    print(f"CLIP Score ($\\uparrow$) & {clip_score_wm:.2f} & Close to {clip_score_no_wm:.2f} (no watermark) \\\\")
print("\\hline")
print("\\end{tabular}")
print("\\end{table}")

print("\n✅ 隐蔽性分析完成！")