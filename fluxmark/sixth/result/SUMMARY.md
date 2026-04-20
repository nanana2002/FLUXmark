# OrthoFlow 实验结果汇总报告

> **实验名称**: OrthoFlow_Topic_0327_HighPerf
> **生成时间**: 2026-04-20 21:42:07
> **样本数量**: 20 张
> **模型路径**: `/home/daiyn/project_flux/model/flux-schnell`
> **输出目录**: `/home/daiyn/project_flux/fluxmark/sixth`

---

## 一、隐蔽性分析 (Invisibility)

- **FID**: `205.91` (lower is better, <50 excellent, <100 good, <200 moderate)
- **CLIP Score (无水印)**: `None`
- **CLIP Score (水印图)**: `None`

## 二、全局鲁棒性分析 (Robustness)

| 攻击类型 | Mean Cosine | Retention Rate | AUC | TPR@0.1%FPR |
|----------|-------------|----------------|-----|-------------|
| blur_0.5 | 0.0547 | 85.9% | 1.0000 | 1.0000 |
| blur_1.0 | 0.0409 | 64.3% | 1.0000 | 1.0000 |
| blur_2.0 | 0.0220 | 34.5% | 1.0000 | 1.0000 |
| blur_3.0 | 0.0141 | 21.9% | 0.9975 | 0.9500 |
| blur_4.0 | 0.0076 | 11.9% | 0.9775 | 0.5500 |
| brightness_0.8 | 0.0587 | 92.0% | 1.0000 | 1.0000 |
| brightness_1.2 | 0.0590 | 92.6% | 1.0000 | 1.0000 |
| contrast_0.8 | 0.0605 | 94.8% | 1.0000 | 1.0000 |
| contrast_1.2 | 0.0595 | 92.8% | 1.0000 | 1.0000 |
| crop_0.50 | 0.0008 | 1.5% | 0.5400 | 0.1000 |
| crop_0.75 | 0.0013 | 2.1% | 0.6050 | 0.0500 |
| jpeg_10 | 0.0045 | 7.2% | 0.8125 | 0.3500 |
| jpeg_20 | 0.0100 | 16.0% | 0.9625 | 0.7500 |
| jpeg_30 | 0.0133 | 21.3% | 0.9950 | 0.9000 |
| jpeg_50 | 0.0196 | 31.1% | 0.9975 | 0.9500 |
| jpeg_75 | 0.0286 | 45.1% | 1.0000 | 1.0000 |
| noise_0.03 | 0.0245 | 38.7% | 1.0000 | 1.0000 |
| noise_0.05 | 0.0173 | 27.0% | 1.0000 | 1.0000 |
| noise_0.10 | 0.0118 | 18.5% | 0.9775 | 0.7500 |
| noise_0.15 | 0.0093 | 14.7% | 0.9700 | 0.8000 |
| noise_0.20 | 0.0060 | 9.5% | 0.8700 | 0.4500 |
| resize_0.25 | 0.0228 | 35.5% | 1.0000 | 1.0000 |
| resize_0.50 | 0.0413 | 64.7% | 1.0000 | 1.0000 |
| resize_0.75 | 0.0548 | 85.9% | 1.0000 | 1.0000 |
| sdedit_0.3 | 0.0105 | 16.5% | 0.9900 | 0.8500 |
| sdedit_0.4 | 0.0062 | 9.8% | 0.9350 | 0.4000 |
| sdedit_0.5 | 0.0059 | 9.4% | 0.9150 | 0.4500 |
| sdxl_style_oil_painting | 0.0146 | 21.8% | 0.8975 | 0.5500 |
| sdxl_style_sketch | 0.0106 | 15.9% | 0.9025 | 0.5000 |

### 关键发现
- **AUC = 1.0 的攻击**: jpeg_75, blur_0.5, blur_1.0, blur_2.0, noise_0.03, noise_0.05, resize_0.75, resize_0.50, brightness_0.8, brightness_1.2, contrast_0.8, contrast_1.2, resize_0.25
- **SDEdit 0.3 (扩散再生)**: AUC=0.9900, TPR@0.1%FPR=0.85 — 这是最难的攻击，表现优异
- **Crop 0.50 (结构性弱点)**: AUC=0.5400 — 这是 VAE + DiT 导致的已知局限性

## 三、篡改定位分析 (Tamper Localization)

| 攻击类型 | Patch-AUC | F1-Score | IoU |
|----------|-----------|----------|-----|
| black_block_center | 0.8967 | 0.7831 | 0.6451 |
| black_block_random | 0.8663 | 0.6936 | 0.5343 |
| copy_move | 0.9253 | 0.6347 | 0.4663 |
| fluxfill_center | 0.7556 | 0.6559 | 0.4907 |
| fluxfill_random | 0.7529 | 0.5930 | 0.4280 |
| splicing | 0.9129 | 0.6133 | 0.4446 |

## 四、消融实验分析 (Ablation Study)

*未找到消融实验数据*


## 五、网格分辨率消融 (Grid Size Ablation)

*未找到网格分辨率消融数据*


## 六、原始结果文件索引

- ✅ **鲁棒性分析 (JSON)**: `/home/daiyn/project_flux/fluxmark/sixth/result/merge.json`
- ✅ **鲁棒性详情 (JSON)**: `/home/daiyn/project_flux/fluxmark/sixth/result/detail.json`
- ✅ **隐蔽性分析 (JSON)**: `/home/daiyn/project_flux/fluxmark/sixth/result/invisibility_analysis.json`
- ❌ **消融定量分析 (JSON)**: `/home/daiyn/project_flux/fluxmark/sixth/result/ablation_analysis.json`
- ❌ **网格分辨率消融 (JSON)**: `/home/daiyn/project_flux/fluxmark/sixth/result/ablation_grid/grid_ablation_summary.json`
- ❌ **消融可视化 (图片)**: `/home/daiyn/project_flux/fluxmark/sixth/result/ablation_viz/`
- ✅ **篡改热力图 (图片)**: `/home/daiyn/project_flux/fluxmark/sixth/result/pic/tamper_img/`

---

*本报告由 `summarize_all_results.py` 自动生成*
