# OrthoFlow 水印实验系统

基于扩散模型的频域正交水印嵌入与提取实验框架。

## 📋 项目概述

本项目实现了一套完整的水印实验流程，包括：
- **水印嵌入**：FFT频域约束 + 正交投影 + 语义掩码
- **水印提取**：VAE编码 + Transformer特征提取 + 8x8分块签名
- **鲁棒性测试**：JPEG压缩、高斯模糊、裁剪、噪声、缩放、亮度调整、黑色方块篡改等
- **篡改定位**：差分热力图 + 自适应阈值
- **消融实验**：验证FFT约束、正交投影、语义绑定的贡献

## 📁 文件结构

```
2026/code/
├── config.json                     # 全局配置文件
├── no_watermark_img_generate.py    # 1. 生成无水印基准图像
├── watermarked_img_generate.py     # 2. 生成带水印图像
├── extract_watermark.py            # 水印提取模块（可被其他脚本调用）
├── attack_img_generate.py          # 3. 生成攻击图像
├── analyze_robustness_result.py    # 4. 鲁棒性分析
├── analyze_invisibility_result.py  # 5. 隐蔽性分析
├── ablation_fft.py                 # 6.1 消融实验：无FFT约束
├── ablation_obj.py                 # 6.2 消融实验：无正交投影
├── ablation_sem.py                 # 6.3 消融实验：语义绑定
├── test.sh                         # 一键运行脚本（交互式）
├── test_go.sh                      # 全自动运行脚本（非交互式，适合批量）
└── README.md                       # 本文件
```

## 🔧 环境要求

### 必需依赖
```bash
# PyTorch (建议 CUDA 11.8+)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Diffusers 和 Transformers
pip install diffusers transformers accelerate

# 其他依赖
pip install numpy pillow tqdm scikit-image matplotlib

# 数据集加载
pip install datasets

# 可选：LPIPS（用于感知距离计算）
pip install lpips

# 可选：sklearn（用于AUC等指标计算）
pip install scikit-learn
```

### 模型准备
确保已下载 FLUX 模型到本地路径（在 `config.json` 中配置）：
```
/data/daiyina/project_flux/model/flux-schnell/
```

## ⚙️ 配置说明

编辑 `config.json` 文件配置实验参数：

```json
{
  "experiment_name": "OrthoFlow_Topic_0327",  // 实验名称
  "num_inference_steps": 4,                   // 扩散步数
  "guidance_scale": 0.0,                      // CFG强度
  "height": 512,                              // 图像高度
  "width": 512,                               // 图像宽度
  "alpha": 0.6,                               // 水印注入强度
  "fft_radius_inner": 2,                      // FFT带通内半径
  "fft_radius_outer": 14,                     // FFT带通外半径
  "mask_region": [6, 26, 6, 26],              // 语义掩码区域[y1,y2,x1,x2]
  "secret_key": 42,                           // 随机种子（密钥）
  "num_samples": 20,                          // 生成样本数量
  "prompt_seed": 42,                          // prompt随机种子
  "model_path": "/data/daiyina/project_flux/model/flux-schnell",
  "hf_cache": "/data/daiyina/hf_cache",
  "output_base_dir": "/data/daiyina/project_flux/first"
}
```

### 关键参数说明

| 参数 | 说明 | 建议值 |
|:---|:---|:---|
| `alpha` | 水印强度，越大水印越强但可能影响画质 | 0.4-0.8 |
| `fft_radius_inner` | FFT低频截止，保留太低频易受色块影响 | 2 |
| `fft_radius_outer` | FFT高频截止，太高频易被JPEG抹除 | 14 |
| `mask_region` | 水印注入区域（32x32 latent中的坐标） | [6,26,6,26] |
| `secret_key` | 水印密钥，决定密码本W的唯一性 | 任意整数 |

## 🚀 使用指南

### 方式一：全自动运行（推荐，适合批量实验）

使用 `test_go.sh` 脚本**全自动**运行所有步骤，无需交互：

```bash
cd 2026/code
chmod +x test_go.sh
./test_go.sh
```

**特点**：
- ✅ 无需交互，自动执行所有步骤
- ✅ 适合大规模实验（如100张图）
- ✅ 自动记录日志和耗时
- ✅ 错误时询问是否继续

**配置样本数量**：编辑 `config.json` 修改 `num_samples`：
```json
{
  "num_samples": 100,  // 改为需要的数量
  ...
}
```

### 方式二：交互式运行

使用 `test.sh` 脚本交互式运行，每步询问是否执行：

```bash
cd 2026/code
chmod +x test.sh
./test.sh
```

### 方式二：分步运行

#### 步骤 1：生成无水印图像（基准）
```bash
python3 no_watermark_img_generate.py
```
- 从 MS-COCO 或 Stable-Diffusion-Prompts 数据集获取 prompts
- 生成无水印图像作为对比基准
- 输出：`/data/daiyina/project_flux/first/pic/no_watermarked_img/`

#### 步骤 2：生成水印图像
```bash
python3 watermarked_img_generate.py
```
- 使用 FFT + 正交投影生成带水印图像
- 提取并保存 8x8 签名 `S_orig`
- 输出：`/data/daiyina/project_flux/first/pic/watermarked_img/`

**注意**：此步骤较耗时，需要显存约 12-16GB。

#### 步骤 3：生成攻击图像
```bash
python3 attack_img_generate.py
```
支持的攻击类型：

**传统攻击**：
- **JPEG压缩**：质量 75, 50, 30, 20
- **高斯模糊**：sigma 0.5, 1.0, 2.0
- **中心裁剪**：比例 0.75, 0.50
- **高斯噪声**：std 0.03, 0.05, 0.10
- **缩放**：比例 0.75, 0.50, 0.25
- **亮度/对比度调整**
- **SDEdit**：加噪后VAE解码（模拟扩散再生）

**篡改定位攻击**：
- **黑色方块**：中心、随机位置、角落（含真实mask）

**深度学习语义修改攻击**（需额外模型）：
- **SDXL I2I**：油画风格、水彩风格、素描风格
  - 模型路径：`/data/daiyina/project_flux/model/sdxl-instructpix2pix`
- **FluxFill Inpaint**：中心区域重绘
  - 模型路径：`/data/daiyina/project_flux/model/flux-fill`

输出：`/data/daiyina/project_flux/first/pic/attack_watermarked_img/{attack_name}/`

#### 步骤 4：提取水印签名（GPU）
```bash
python3 extract_watermarks.py
```
- 加载 FLUX 模型到 GPU
- 提取所有攻击图像的签名 S_tamp
- 提取原始图像签名 S_orig 和无水印图像签名 S_no_wm
- **输出**：保存到 `watermark_extra/` 目录

**注意**：此步骤需要 GPU，但一次性运行，之后可以重复分析而不需要重新提取

#### 步骤 5：鲁棒性分析（纯CPU）
```bash
python3 analyze_robustness_result.py
```
- **纯分析脚本，不加载模型，只读取预计算的签名文件**
- 计算指标：
  - **全局鲁棒性**：Mean Cosine、Retention Rate、AUC-ROC、TPR@0.1%FPR
  - **篡改定位**：Patch-AUC、F1-Score、IoU

**优势**：分析阶段不需要 GPU，可以快速重复运行

#### 步骤 5：隐蔽性分析
```bash
python3 analyze_invisibility_result.py
```
对比水印图像与无水印图像：
- **PSNR**（峰值信噪比）：>40dB 优秀
- **SSIM**（结构相似性）：>0.99 优秀
- **LPIPS**（感知距离）：<0.02 优秀

生成结果：`result/invisibility_analysis.json`

#### 步骤 6：消融实验（可选）
```bash
# 6.1 无FFT约束
python3 ablation_fft.py

# 6.2 无正交投影
python3 ablation_obj.py

# 6.3 语义绑定
python3 ablation_sem.py
```

### 方式三：使用提取模块单独提取水印

```python
from extract_watermark import WatermarkExtractor, extract_signature_from_file

# 方式1：使用类
extractor = WatermarkExtractor()
S = extractor.extract_signature(
    image="path/to/image.png",
    prompt="a cat sitting on a table"
)
stats = extractor.compute_core_stats(S)
print(f"核心区域均值: {stats['mean']:.4f}")

# 方式2：命令行
python3 extract_watermark.py -i image.png -o signature.npy -p "prompt text"
```

## 📊 输出结果说明

### 目录结构
```
/data/daiyina/project_flux/first/
├── prompts_used.json                 # 实际使用的prompts
│
├── pic/
│   ├── no_watermarked_img/           # 无水印图像
│   │   ├── no_wm_000.png
│   │   ├── no_wm_001.png
│   │   └── ...
│   │
│   ├── watermarked_img/              # 水印图像
│   │   ├── wm_000.png
│   │   ├── wm_000_signature.npy      # 8x8签名S_orig
│   │   ├── wm_001.png
│   │   └── ...
│   │
│   ├── attack_watermarked_img/       # 攻击后图像
│   │   ├── jpeg_50/
│   │   │   ├── wm_000.jpg
│   │   │   └── ...
│   │   ├── blur_1.0/
│   │   ├── black_block_center/       # 篡改攻击含mask
│   │   │   ├── wm_000.png
│   │   │   └── wm_000_mask.npy       # 真实篡改掩码
│   │   └── ...
│   │
│   ├── ablation_fft_watermarked_img/ # 消融实验结果
│   ├── ablation_obj_watermarked_img/
│   └── ablation_sem_watermarked_img/
│
└── result/
    ├── detail.json                   # 详细分析结果
    ├── merge.json                    # 汇总表格数据
    ├── invisibility_analysis.json    # PSNR/SSIM/LPIPS
    └── pic/
        └── tamper_img/               # 篡改热力图PNG
            ├── black_block_center_wm_000_heatmap.png
            └── ...
```

### 结果文件格式

**merge.json**（鲁棒性汇总）：
```json
{
  "jpeg_50": {
    "mean_cosine": 0.8234,
    "std_cosine": 0.0456,
    "mean_retention_rate": 95.2,
    "auc": 0.9876,
    "tpr_at_0.1%_fpr": 0.9234
  },
  "black_block_center": {
    "mean_cosine": 0.6543,
    "mean_patch_auc": 0.8765,
    "mean_f1": 0.7654,
    "mean_iou": 0.5432
  }
}
```

**signature.npy**（签名文件）：
- 形状：`(8, 8)`
- 值域：`[-1, 1]`，越高表示水印匹配度越高
- 正常值：0.15-0.35（核心区域均值）

## 🔍 结果解读

### 鲁棒性指标

| 指标 | 说明 | 优秀标准 |
|:---|:---|:---|
| Mean Cosine | 攻击后签名与原签名的余弦相似度 | >0.85 |
| Retention Rate | 信号保留率 | >90% |
| AUC | ROC曲线下面积 | >0.95 |
| TPR@0.1%FPR | 万分之一误报率下的检出率 | >0.90 |

### 篡改定位指标

| 指标 | 说明 | 优秀标准 |
|:---|:---|:---|
| Patch-AUC | 块级AUC | >0.85 |
| F1-Score | 精确率和召回率的调和平均 | >0.75 |
| IoU | 预测与真实篡改区域交并比 | >0.60 |

### 隐蔽性指标

| 指标 | 说明 | 优秀标准 |
|:---|:---|:---|
| PSNR | 峰值信噪比 | >40 dB |
| SSIM | 结构相似性 | >0.99 |
| LPIPS | 感知距离 | <0.02 |

## ⚠️ 常见问题

### 1. 显存不足
**问题**：运行时报错 `CUDA out of memory`

**解决**：
- 减少 `num_samples`（如从20减到10）
- 确保已卸载Text Encoder（代码已自动处理）
- 使用更小batch size（目前为1）

### 2. 模型加载失败
**问题**：`Model not found` 或无法下载

**解决**：
- 检查 `config.json` 中的 `model_path` 是否正确
- 确保模型文件已下载到本地路径
- 检查 HuggingFace token（如需要）

### 3. 数据集加载失败
**问题**：无法加载 MS-COCO 或 SD-Prompts

**解决**：
- 代码会自动回退到默认prompts
- 或手动修改 `no_watermark_img_generate.py` 使用本地prompts

### 4. 提取签名值过低
**问题**：核心区域均值 < 0.1

**可能原因**：
- `secret_key` 不匹配（检查config.json）
- 图像未加水印
- VAE/Transformer加载错误

### 5. 篡改定位效果差
**问题**：IoU < 0.3

**可能原因**：
- 攻击强度过大（如黑色方块太小）
- 自适应阈值不合适（可调整 `threshold_ratio`）
- 全局注意力泄露（正常现象，见理论文档）

## 📝 实验流程建议

### 快速验证（约1小时）
```bash
# 1. 修改config.json
"num_samples": 5  # 只生成5张图

# 2. 运行核心流程
python3 no_watermark_img_generate.py
python3 watermarked_img_generate.py
python3 attack_img_generate.py
python3 analyze_robustness_result.py
```

### 完整实验（约4-6小时，20张图）
```bash
./test.sh
# 按提示依次选择 y 运行所有步骤
```

### 大规模实验（约8-12小时，100张图）
```bash
# 1. 修改config.json
"num_samples": 100  # 改为100张
"experiment_name": "OrthoFlow_Scale_100"  # 修改实验名称

# 2. 使用全自动脚本（无需交互）
./test_go.sh

# 3. 脚本会自动完成所有步骤并输出报告
```

**大规模实验特点**：
- 样本数量多，统计结果更稳定
- 可以分析标准差和置信区间
- 适合论文中的大规模验证

**存储需求估算**（100张图）：
- 无水印图像：~100 MB
- 水印图像：~100 MB
- 攻击图像（~25种攻击）：~2.5 GB
- 签名文件：~50 MB
- **总计约：2.7 GB**

### 论文复现
1. 运行完整实验（建议50-100张图，所有攻击）
2. 对比消融实验结果
3. 提取 `merge.json` 中的数据制作表格
4. 使用 `result/pic/tamper_img/` 中的热力图

## 📚 理论参考

详见 `2026/conclusion/topic_0327.md`，关键概念：
- **FFT带通约束**：保留中频段，抗压缩
- **正交流形注入**：正交投影保证画质
- **差分签名定位**：`S_orig - S_tamp` 热力图

## 🐛 调试技巧

### 检查签名是否正常
```python
import numpy as np
S = np.load('wm_000_signature.npy')
print(f"签名形状: {S.shape}")
print(f"核心区域均值: {S[1:7, 1:7].mean():.4f}")
print(f"整体范围: [{S.min():.4f}, {S.max():.4f}]")
```

### 可视化签名
```python
import matplotlib.pyplot as plt
plt.imshow(S, cmap='hot')
plt.colorbar()
plt.title('8x8 Watermark Signature')
plt.savefig('signature_vis.png')
```

## 📧 联系方式

如有问题，请检查：
1. 配置文件路径是否正确
2. 显存是否充足
3. 模型文件是否完整
4. 依赖包是否安装

---

**最后更新**：2026-03-27
