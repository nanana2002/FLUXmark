# OrthoFlow 规模化实验框架

## 文件说明

| 文件 | 用途 |
|------|------|
| `prompts.json` | 实验配置文件，包含20条prompt和超参数 |
| `batch_generate.py` | 批量生成20张水印图像 |
| `batch_test.py` | 规模化鲁棒性测试（每种攻击5次） |
| `analyze_results.py` | 结果分析和可视化 |

## 快速开始

### 1. 配置实验参数

编辑 `prompts.json`：
```json
{
  "experiment_name": "OrthoFlow_Test_20",
  "prompts": ["...", "..."],  // 20条提示词
  "alpha": 0.6,              // 注入强度
  "fft_radius_inner": 2,     // FFT内半径
  "fft_radius_outer": 14,    // FFT外半径
  "mask_region": [6, 26, 6, 26]  // 水印区域
}
```

### 2. 批量生成水印图像

```bash
cd /data/daiyina/project_flux
python batch_generate.py
```

输出：
- `img_000.png` ~ `img_019.png` (20张图像)
- `img_000_signature.npy` ~ `img_019_signature.npy` (签名)
- `OrthoFlow_Test_20_summary.json` (生成摘要)

### 3. 规模化鲁棒性测试

```bash
python batch_test.py
```

测试的攻击类型（每种跑5次）：
| 攻击 | 参数 |
|------|------|
| baseline | 无攻击 |
| jpeg_75/50/30/20 | JPEG压缩质量 |
| blur_0.5/1.0/2.0/3.0 | 高斯模糊sigma |
| crop_0.75/0.50 | 中心裁剪比例 |
| noise_0.03/0.05 | 噪声标准差 |
| resize_0.75/0.50 | 缩放比例 |

输出：
- `OrthoFlow_Test_20_test_results.json` (完整测试结果)

### 4. 分析结果

```bash
# 查看最新实验结果
python analyze_results.py

# 查看指定实验
python analyze_results.py OrthoFlow_Test_20

# 对比多个实验
python analyze_results.py Exp1 Exp2 Exp3
```

输出包括：
- 汇总表格（均值、标准差、中位数、四分位数）
- 逐图像分析
- LaTeX表格代码
- CSV格式数据

## 结果解读

### 判定标准
| 相似度 | 状态 | 说明 |
|--------|------|------|
| > 0.85 | ✅ 强 | 水印强保留，可靠检测 |
| 0.70-0.85 | 🟢 良 | 水印良好保留 |
| 0.55-0.70 | 🟡 中 | 水印中度保留 |
| 0.40-0.55 | 🟠 弱 | 水印弱保留 |
| < 0.40 | 🔴 差 | 水印可能丢失 |

### 关键指标
- **Mean**: 20张图 × 5次运行的平均相似度
- **Std**: 标准差，衡量稳定性
- **Retention**: 相对于无攻击图像的保留率
- **Q25/Q75**: 四分位数，反映数据分布

## 调参建议

如果鲁棒性不够：
```json
{
  "alpha": 0.8,              // 提高注入强度 (0.6→0.8)
  "fft_radius_inner": 1,     // 扩大低频范围 (2→1)
  "fft_radius_outer": 15     // 扩大高频范围 (14→15)
}
```

如果图像质量下降：
```json
{
  "alpha": 0.4,              // 降低注入强度
  "fft_radius_inner": 3,     // 缩小低频范围
  "fft_radius_outer": 12     // 缩小高频范围
}
```

## 实验记录示例

```bash
# 实验1: 默认参数
python batch_generate.py
python batch_test.py
mv pic/OrthoFlow_Test_20_test_results.json results/exp_baseline.json

# 实验2: 高注入强度
# 修改 prompts.json: alpha=0.8
python batch_generate.py
python batch_test.py
mv pic/OrthoFlow_Test_20_test_results.json results/exp_high_alpha.json

# 对比实验
python analyze_results.py exp_baseline exp_high_alpha
```
