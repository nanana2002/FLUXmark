# OrthoFlow：面向 Rectified Flow 的免训练隐空间水印

> **专为 FLUX / SD3 等基于 Rectified Flow 的 Diffusion Transformers (DiT) 设计的单步提取隐空间水印方案。**

---

## 1. 核心创新

| 模块 | 说明 |
|------|------|
| **Rectified Flow 强绑定** | 充分利用流匹配模型“直线轨迹”的数学特性，实现单步高效提取。旧时代基于弯曲 DDPM 轨迹的模型（SD1.5/2.x）不适用，这不是缺点，而是拥抱新架构的范式转移。 |
| **频域 FFT 约束** | 在 32×32 的频域空间中锁定中频段 `[2, 14]`，既避开极低频（避免色块），又避开高频（避免被 JPEG/高斯模糊抹除），极大提升水印抗打击能力。 |
| **Token 级正交流形注入 (Orthogonal Drift)** | 在注入时将水印能量投影到 latent 速度场的零空间（Null Space），保证**绝对不毁画质**。 |
| **全图统一注入 (Edge-to-Edge Global)** | 得益于正交投影，我们彻底抛弃了传统水印为保画质而使用的局部 `mask_region`，实现真正的全画幅保护。背景角落的篡改也能被完美检测。 |
| **差分签名定位 (Differential Signature)** | 保存原始签名 `S_orig` 作为“防伪档案”，篡改检测时通过 `|S_orig - S_tamp|` 画出零误报热力图。 |
| **8×8 最优网格 (Sweet Spot)** | 在 VAE 8 倍压缩的物理极限下，32×32（64D）方差太大，16×16（256D）仍不够稳定，8×8（1024D）是高维统计学稳定性与空间精度的最优解。 |

---

## 2. 项目结构

```
.
├── config.json                     # 实验配置（模型路径、alpha、密钥等）
├── prompts.json                    # 200 条生成 Prompt（MS-COCO + SD-Prompts）
├── prompts_modified.json           # LLM 生成的语义修改 Prompt（攻击用）
│
├── generate_prompt.py              # Step 0: 生成 prompts.json（只需执行一次）
├── no_watermark_img_generate.py    # Step 1: 生成无水印基准图
├── watermarked_img_generate.py     # Step 2: 生成带水印图像 + 提取 S_orig
├── attack_img_generate.py          # Step 3: 对水印图施加多种攻击
├── extract_watermarks.py           # Step 4: 批量提取所有图像的 8×8 签名
├── analyze_robustness_result.py    # Step 5: 鲁棒性 & 篡改定位分析
├── analyze_invisibility_result.py  # Step 6: FID / CLIP Score 隐蔽性评估
│
├── ablation_fft.py                 # 消融：不加 FFT 约束
├── ablation_obj.py                 # 消融：不加正交投影
├── ablation_obj_stress.py          # 消融：不加正交投影（alpha=2.5 压力测试）
├── ablation_sem.py                 # 消融：使用语义绑定掩码（旧方案对比）
├── ablation_grid.py                # 消融：网格分辨率 32×32 / 16×16 / 8×8
├── analyze_ablation.py             # 消融实验综合分析（FID/CLIP/S_mean）
├── ablation_attack_pipeline.py     # 一体化消融攻击+提取+可视化脚本
│
└── test.sh                         # 完整实验流程一键串联脚本
```

---

## 3. 快速开始

### 3.1 环境准备

需要安装 `diffusers`、`torch`、`torchmetrics`、`opencv-python`、`scikit-learn` 等依赖：

```bash
# 基础环境（生成 + 提取）
pip install -r requirements_fluxenv.txt

# 攻击环境（SDXL / FluxFill）
pip install -r requirements_attackenv.txt
```

### 3.2 配置修改

编辑 `config.json`，修改以下关键路径为你服务器上的实际路径：

```json
{
  "model_path": "/data/daiyina/project_flux/model/flux-schnell",
  "sdxl_path": "/data/daiyina/project_flux/model/sdxl-instructpix2pix",
  "flux_fill_path": "/data/daiyina/project_flux/model/flux-fill",
  "output_base_dir": "/data/daiyina/project_flux/sixth",
  "gpu_id": 0
}
```

### 3.3 运行完整流程

```bash
# 交互式运行（会逐步询问）
bash test.sh

# 或手动分步执行：
python3 generate_prompt.py              # 只需一次
python3 no_watermark_img_generate.py    # 1. 无水印基准图
python3 watermarked_img_generate.py     # 2. 水印图 + S_orig
python3 attack_img_generate.py          # 3. 攻击图像
python3 extract_watermarks.py           # 4. 提取所有签名
python3 analyze_robustness_result.py    # 5. 鲁棒性分析
python3 analyze_invisibility_result.py  # 6. 隐蔽性分析
```

---

## 4. 实验流程详解

### Phase 1: 图像生成
- `no_watermark_img_generate.py`：批量生成 200 张无水印 FLUX 图像，作为 FID/CLIP 的基准。
- `watermarked_img_generate.py`：
  1. 预编码所有 Prompt 并常驻 GPU；
  2. 生成 FFT 密码本 `W`（全图统一）；
  3. 通过 `orthogonal_drift_callback` 在每个 timestep 做 Token 级正交投影注入；
  4. 生成完成后立即用 VAE + Transformer 单步提取 `v_pred`，计算并保存 `8×8` 签名 `S_orig`。

### Phase 2: 攻击生成
`attack_img_generate.py` 支持以下攻击类型：

| 类别 | 具体攻击 |
|------|---------|
| **传统信号攻击** | JPEG 压缩、高斯模糊、中心裁剪、高斯噪声、缩放、亮度/对比度调整 |
| **篡改定位攻击** | 黑色方块（中心/随机）、**Copy-Move**、**Splicing** |
| **深度学习攻击** | SDXL 全局风格迁移、FluxFill 局部重绘、SDEdit 扩散再生 |

- **Copy-Move**：从图像内部复制一个块粘贴到另一位置。
- **Splicing**：从同批次随机另一张图中取块拼接到当前图。
- 深度学习攻击支持自动读取 `prompts_modified.json` 中的 LLM 生成 Prompt。

### Phase 3: 签名提取
`extract_watermarks.py`：
- 批量提取 `S_orig`（原始水印图）、`S_tamp`（攻击后图）、`S_no_wm`（无水印负样本）。
- **新增**：自动检测 `{output_base_dir}/pic/real_images/` 目录中的真实自然图像，提取 `_S_real.npy` 作为额外负样本，用于更严谨的 TPR@FPR 计算。

### Phase 4: 分析评估
- `analyze_robustness_result.py`：
  - **全局鲁棒性**：Mean Cosine、Retention Rate、AUC-ROC、**TPR@0.1%FPR**（使用无水印图 + 真实图像作为负样本）。
  - **篡改定位**：Patch-AUC、F1-Score、IoU，并自动生成高分辨率篡改热力图。
- `analyze_invisibility_result.py`：计算水印图与无水印图之间的 FID 和 CLIP Score。

---

## 5. 消融实验

### 5.1 已有消融

```bash
python3 ablation_fft.py       # w/o FFT：验证频域约束必要性
python3 ablation_obj.py       # w/o Orthogonal：验证正交投影必要性（常规 alpha）
python3 ablation_sem.py       # w/ Semantic Mask：旧方案局部掩码对比
```

### 5.2 新增消融

**正交投影压力测试（Stress Test）**：
```bash
python3 ablation_obj_stress.py
```
- 将 `alpha` 拉爆到 `2.5`（正常为 `0.6`）。
- 没有正交投影时图像会变成扭曲怪兽（FID 暴涨），而有正交时依然完美。
- 极具冲击力地证明：**正交投影极大抬高了水印注入强度的天花板（Upper Bound）**。

**网格分辨率消融（Grid Size Ablation）**：
```bash
python3 ablation_grid.py
```
- 对已有水印图分别提取 `32×32`（64D）、`16×16`（256D）、`8×8`（1024D）三种粒度签名。
- 输出统计结果到 `{output_base_dir}/result/ablation_grid/grid_ablation_summary.json`。
- 预期结论：`32×32` 方差极大假阳性飙升，`16×16` 有所改善但不够稳定，`8×8` 是最优解。

**消融综合分析**：
```bash
python3 analyze_ablation.py
```
- 对比完整版与各种消融版本的 FID、CLIP Score、Signature Mean，并自动生成 LaTeX 表格。

---

## 6. 结果输出结构

实验结果默认输出到 `config.json` 中配置的 `output_base_dir`（如 `/data/daiyina/project_flux/sixth`）：

```
third/
├── prompts.json
├── prompts_modified.json
├── prompts_modified_used.json      # 攻击脚本自动记录实际使用的修改 Prompt
│
├── pic/
│   ├── no_watermarked_img/         # 无水印基准图
│   ├── watermarked_img/            # 水印图 + S_orig 签名
│   ├── attack_watermarked_img/     # 各类攻击图像
│   │   ├── jpeg_50/
│   │   ├── blur_1.0/
│   │   ├── crop_0.50/
│   │   ├── black_block_center/
│   │   ├── copy_move/              # 手工篡改
│   │   ├── splicing/               # 手工篡改
│   │   ├── sdxl_style_oil_painting/
│   │   ├── fluxfill_center/
│   │   └── sdedit_0.3/
│   ├── ablation_fft_watermarked_img/
│   ├── ablation_obj_watermarked_img/
│   ├── ablation_obj_stress_watermarked_img/
│   ├── ablation_sem_watermarked_img/
│   └── real_images/                # 手动放入真实自然图像（负样本）
│
├── watermark_extra/                # 所有 npy 签名文件
│   ├── wm_xxx_S_orig.npy
│   ├── no_wm_xxx_S_no_wm.npy
│   ├── real_xxx_S_real.npy
│   └── wm_xxx_attack_name_S_tamp.npy
│
└── result/
    ├── merge.json                  # 鲁棒性汇总表格数据
    ├── detail.json                 # 每张图/每种攻击的详细结果
    ├── invisibility_analysis.json  # FID / CLIP Score
    ├── ablation_analysis.json      # 消融综合结果
    ├── pic/
    │   └── tamper_img/             # 篡改定位热力图
    └── ablation_grid/
        └── grid_ablation_summary.json
```

---

## 7. 关键学术论点（对应论文写作）

1. **为什么只支持 FLUX/SD3？**  
   因为 Rectified Flow 的直线轨迹是单步提取的数学基石。旧模型（SD1.5/2.x）基于弯曲 DDPM 轨迹，强行适配会导致巨大的泰勒展开截断误差。我们拥抱新架构。

2. **为什么热力图只有 8×8？**  
   这不是阈值问题，而是 VAE 8 倍压缩 + DiT Patch 化的**物理分辨率瓶颈**。512×512 像素经 VAE 后潜空间仅 32×32，再聚合为 8×8 已是极限。8×8 对应 1024 维余弦相似度向量，底噪被极度压缩，是统计学稳定与空间精度的最优权衡。

3. **为什么不再需要语义 Mask？**  
   正交投影在数学上保证了水印能量不会干扰任何语义区域。既然全图注入都不毁画质，为什么还要把水印局限在中心？局部 Mask 会导致背景篡改完全检测不到（0−0=0），全图注入是正交流形的必然选择。

4. **如何应对 Crop 攻击的脆弱性？**  
   在 Limitations 中坦诚面对：这是 **VAE Convolutional Pollution** 与 **DiT Global Attention Shift** 共同导致的不可逆污染，是所有隐空间网格水印的结构性痛点（Achilles' Heel）。隐藏失败是学术大忌，用高维原理解释失败是加分项。

---

## 8. 注意事项

- **显存要求**：脚本针对 ~50GB 显存优化（模型常驻 GPU、批量推理）。若显存较小，可适当调低 `batch_size`。
- **真实图像负样本**：如需计算更严谨的 TPR@FPR，请将自然实拍照片放入 `{output_base_dir}/pic/real_images/`，再运行 `extract_watermarks.py`。
- **prompts_modified.json**：SDXL 风格迁移和 FluxFill 局部重绘会自动读取该文件中的 LLM 生成 Prompt。若未提供，将回退到默认 Prompt。

---

## 9. 作者与致谢

本项目基于 Rectified Flow 的数学特性，探索 Diffusion Transformers 时代下的免训练隐空间水印新范式。欢迎讨论与交流！
