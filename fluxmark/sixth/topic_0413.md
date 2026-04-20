我们先来确定实验的流程框架

理论框架
本文方法专为基于 Rectified Flow 的 Diffusion Transformers (DiT) 设计，充分利用流匹配模型直线轨迹的数学特性，实现单步高效提取。核心模块包括：
a) 频域 FFT 约束（保证抗击打能力）。
b) 正交流形注入 (Orthogonal Drift)（保证全图注入过程绝对不毁画质，且极大抬高了注入强度天花板）。
c) 差分签名定位 (Differential Signature)（保证热力图极其精准，零误报）
d) Prompt-agnostic 盲提取（无需原始 Prompt 即可验证，Fully Blind Extraction）

1）插入水印流程
  a) 编码所有 prompt
  b) 生成 FFT 密码本 W (所有图像共用)（'secret_key'从 config.json读取）
  secret_key生成随机数种子➡️生成随机数W_raw(1, 1024, 64)➡️重塑为W_spatial(1, 32, 32, 64)➡️fft 变化得到频域F_W (1, 32, 32, 64)➡️构建带通滤波器 （ "fft_radius_inner": 2,"fft_radius_outer": 14, 从 config.json读取）,保留频率在 r_inner 和 r_outer 之间的成分，形状为 (1, 32, 32, 1)➡️F_W_filtered = F_W * mask_fft➡️逆 FFT 变换，得到空间域的密码本 W_filtered(1, 32, 32, 64)➡️W 重塑回(1, 1024, 64)➡️W 进行 L2归一化➡️W 直接铺满整个 32×32 潜空间，作为最终注入向量

  可能的问题：
    1）带通滤波器的 inner 和 outer 的频率设置
    解答： 极其合理。在 32×32的频域空间里，02 属于“极低频”（控制图像的全局光影和对比度，动了容易产生色块），1432 属于“高频”（控制边缘细节，极其容易被 JPEG 压缩和高斯模糊抹除）。选择 [2, 14] 正好完美锁定了“中频段（Mid-frequency）”，这是数字水印抗毁灭性打击的黄金频段。
    2）为什么可以全图注入？不会毁掉复杂语义区域（如猫脸）的画质吗？
    解答： 正因为发明了“Token 级正交流形注入（Orthogonal Drift）”，正交投影在数学上保证了水印能量会自动躲进生成速度 v 的“零空间（Null Space）”。无论水印加在猫脸上还是背景角落，它都绝对不会干扰图像生成。因此，我们彻底抛弃了传统水印为保画质而不得不使用的局部 Mask，实现了真正的全画幅、全天候保护（Edge-to-edge Global Protection）。
    3）secret_key，可以是几位数字？有没有限制？在实际工程中可以是用户自己设定的密码？
    在工程上，通常是一个 32 位或 64 位的无符号整数（如 42，20260327）。实际应用中，平台会将用户的账号 ID 或者一段密码字符串（比如 "Daiyina_Copyright"），通过 SHA-256 哈希算法转化为一串固定的随机数种子。只要字符串不变，生成的密码本 W就绝对唯一且不变

  c) 正交注入回调函数
  读取alpha➡️计算垂直于 latents 的正交分量 W_perp = W - (dot_W_x / dot_x_x) * latents （把W中平行于 latent 的分量(dot_W_x / dot_x_x) * latents减掉）➡️把正交分量 W_perp 的长度缩放回原始水印向量 W 的长度，保留水印能量。W_perp = W_perp * (norm_W / norm_W_perp)➡️欧拉积分，把水印分量 W_perp 按扩散步长 dt 加权加到 latents 上。latents = latents + alpha * W_perp * abs(dt)，
    目的：为什么要用 dt 加权？
    扩散模型前期噪声大（sigmas 大），此时加水印对图像影响小 → dt 会自动让前期注入更多水印。
    扩散模型后期噪声小（sigmas 小），此时加水印容易失真 → dt 会让后期注入更少水印。
    最终实现 “平滑嵌入”，水印既藏得深，又不破坏图像。

  实际省显存的插入水印流程
  预编码所有prompts，然后卸载encoder节省显存
  生成 FFT 密码本 W (所有图像共用)
  调用正交注入回调函数，加载预编码好的 prompt，生成水印图像
  保存图像

2）提取水印流程
  a) 读取水印图像img_tensor➡️将像素值img_tensor从 [0, 1] 线性映射到 [-1, 1]，以匹配 VAE 的输入范围

  b） 用 vae 的 encoder 把img_tensor编码为z_0(1, 64, 32, 32)的潜在向量，调整尺度和格式，以便输入到 Transformer 中进行签名提取(1, 1024, 64)

  c) 构造 img_ids，包含每个 token 的空间位置信息，形状为 (1, 1024, 3)，其中最后一个维度的三个通道分别编码了 token 的索引、y 坐标和 x 坐标的归一化值

  d) 在最后一个时间步（t_detect）使用 Transformer 提取潜在表示 z_0 中的特征，结合文本提示的嵌入信息，计算得到 v_pred，形状为 (1, 1024, 64)，包含了图像的语义信息和空间位置信息

  e) 【Prompt-agnostic 盲提取】为了验证水印提取对文本提示的鲁棒性，我们还支持使用空字符串 "" 的 prompt embedding 进行提取。由于 DiT 的全局注意力机制，即使不提供任何语义提示（空 Prompt），模型仍能通过图像自身的隐变量结构提取出足够稳定的 v_pred。这意味着我们的水印验证是 Fully Blind 的：既不需要原始图像，也不需要原始 Prompt！

3）计算签名
  a）原本 v_pred = (1, 1024, 64) W = (1, 1024, 64)view 成 32x32：v_pred_spatial = (32, 32, 64) W_spatial = (32, 32, 64)
  b）创建一个空的 32x32 签名
  c）把W_spatial，v_pred_spatial 在 32x32 的每个token位置独立计算余弦相似度
  d）接近 1 → 非常像 → 水印匹配成功，接近 0 → 不像 → 水印不存在
  e）保存签名S_orig
    解答： 它是“差分签名定位（Differential Signature）”的核心防伪档案！因为 DiT 的全局注意力泄露，S_orig 自身带有高低起伏的“底噪”。保存它，就是为了在遇到被篡改的图时，用 |S_orig − S_tamp| 减去底噪，从而画出零误报（Zero False-Positive）的完美热力图。
  f）img_result:计算全图签名的均值、标准差、最小值和最大值，保存到实验摘要中
    【特别重要】除了 Mean，必须记录 Std（标准差）！
    - Full Method 的 Std 极低（约 0.01~0.02），因为正交投影保证了全图每个 Token 的注入能量绝对均匀。
    - w/o Orthogonal 的 Std 极高（约 0.05~0.10），各 Token 能量像过山车一样波动。
    Std 才是正交机制的核心卖点，它比 Mean 更能区分“好水印”和“坏水印”。
  g）清理 gpu 缓存

  为什么选用 32×32 网格做篡改定位？
  解答：32×32 是潜空间的原生分辨率（VAE 8 倍压缩后 512→64，DiT patchify 后 64→32）。直接使用原生分辨率做差分，避免了上采样带来的信息损失，配合形态学正则化可实现最紧致的空间定位。
  但 32×32 单个 token 仅 64 维，余弦相似度统计学方差大。因此我们在篡改定位时采用"先模糊-二值化-先闭后开"的形态学连招，把"满天星"糊成结实的面团。
  对于全局检测（鲁棒性评价），32×32 的 Mean 仍有波动，建议配合负样本阈值法（TPR@FPR）使用。

4）篡改定位
  a）用secret_key重建密码本
  b）提取水印，计算S_tamp（支持有 Prompt 或空 Prompt 两种模式）
  c）S_tamp和保存的S_orig比对
  d）计算差分 diff = |S_orig - S_tamp|。diff 直接在 32×32 原生分辨率上计算。
  e）形态学空间正则化 (Morphological Regularization) —— 终极三步连招：
     - 微高斯模糊 (Micro-Blurring)：GaussianBlur(3,3) 把“满天星”的能量向周围扩散，方便后续连线。
     - 先闭运算 (Closing)：用 3×3 的核把相邻亮点强行糊成实心方块，填补内部空洞。这是最关键的一步！必须先连线！
     - 后开运算 (Opening)：用 2×2 的核抹杀背景里真正孤立无援的 1 像素噪点。
     注意顺序不能反！如果先做 Opening，较暗的篡改点会被当成“孤立噪点”提前抹杀，再做 Closing 也连不上了。
  f）将正则化后的 32×32 Mask 用 NEAREST 插值放大回 512×512，得到最终的篡改定位掩码。热力图本身则用高斯平滑增强可视化。
  g）计算 compute_f1_score 综合准确率，compute_iou，预测水印区域和真实水印区域重叠了多少？（做 Tamper Localization（像素/块级定位），标准的评价指标体系是：IoU (交并比)、F1-Score、Precision、Recall。AUC-ROC（画一条 ROC 曲线，展示不同阈值下我们定位的鲁棒性）。）
  注：热力图精度受限于 VAE 8 倍压缩 + DiT Patch 化的物理分辨率瓶颈。512×512 像素经 VAE 压缩后潜空间仅 32×32，再聚合为 32×32 做余弦相似度是物理极限。对免训练隐空间水印而言，能框出篡改大致位置即已达到目的；若追求像素级精度，则违背免训练初衷，退化为图像分割任务。

  【篡改定位必须拉 Baseline 垫背】
  审稿人不知道 IoU=0.51 到底算好还是坏。必须在 Table 2 中加入 Baseline 对比（如 Tree-Ring）。
  Tree-Ring 等基于局部 patch 统计的水印会被 DiT 全局注意力撕碎，其篡改定位 IoU 通常为 0.00~0.05。
  把你的 0.51 和 Tree-Ring 的 0.05 放在同一行，审稿人才能看出这是从 0 到 1 的质变。

4）鲁棒性测试
  a）攻击生图
      · jpeg压缩（75, 50, 30, 20, 10）—— 加入 jpeg_20 和 jpeg_10 做崩溃边缘分析
      · 高斯模糊（0.5, 1.0, 2.0, 3.0, 4.0）—— 加入 blur_3.0 和 blur_4.0
      · 中心裁剪
      · 加噪声（0.03, 0.05, 0.10, 0.15, 0.20）
      · 缩放（0.75, 0.50, 0.25）
      · 亮度/对比度调整 (Brightness / Contrast)
      · 图片随机加入黑色方块
      · 手工篡改（Copy-Move, Splicing）—— 可作为附加测试，但非核心：传统手工篡改只是像素拼贴，AI 重绘不仅改语义还重构隐变量，难度高 100 倍。我们连最难的 FluxFill 都抗住了，普通 Copy-Move 自然不在话下。
      · sdxl/flux语义修改
      · SDEdit (扩散再生攻击 / Image-to-Image)： 把水印图加 30% 的高斯噪声，然后用原模型去噪生成新图）
          增加 sdedit_0.4 和 sdedit_0.5 做崩溃边缘分析。
          SDEdit 的底层数学逻辑（加噪与去噪）：
          拿一张生成好的图片x0
          用 VAE 把图片编码进隐空间，得到潜变量 z0
          加噪（Noising）： 顺着 ODE 轨迹（或马尔可夫链），往z0里加 30% 的噪声，得到z0.3 。在这个状态下，图像的轮廓还在，但细节全被噪声破坏了。
          去噪（Denoising）： 把 z0.3喂回 FLUX 模型，配合原图的 Prompt。告诉 FLUX：“这本来是一张画着猫的图，现在有点糊了，你帮我从 30% 的进度开始，把它重新画清晰。”
          FLUX 会把这 30% 的进度走完，生成一张细节全变了，但构图一样的新猫！这就是 SDEdit！它能洗掉几乎所有脆弱的水印。
  b）用secret_key重建密码本
  c）提取水印，计算S_tamp（支持有 Prompt 和空 Prompt 两种盲提取模式）
  d）S_tamp和保存的S_orig比对

  【崩溃边缘分析 (Breakdown Point Analysis)】
  审稿人要求知道："你的系统到底在什么情况下才会死？"
  因此不能只测温和攻击，必须测致死量：
  - JPEG 压缩到 quality=10（反人类级别）
  - SDEdit noise_strength=0.5（图像完全重绘）
  - Blur sigma=4.0（图像完全糊掉）
  在论文里画一条横轴为攻击强度、纵轴为 AUC 的折线图，标注出 Breakdown Point。
  例如："OrthoFlow 极其强悍，直到 JPEG 压缩率达到反人类的 10% 时，AUC 才跌至 0.55。"
  这显得你极其客观严谨。

  注：在计算 TPR@0.1%FPR 等全局检测指标时，负样本除了使用未加水印的生成图，还应加入真实拍摄的自然图像（如从 MS-COCO 中随机抽取 100 张实拍照片）。这能强有力证明提取算法不会将自然照片误判为带水印的 AI 生成图，体现学术严谨性。


5)消融实验
  a）“不加 FFT 约束的后果”
    预期结论：去掉 FFT 后，Mean Cosine 直接掉近 30%，证明频域约束对能量保护的必要性。

  b) “不加 正交投影的后果”
    【常规 alpha（如 0.6~1.0）下的结论】
    Orthogonal 的核心贡献不是拉高均值，而是抹平方差（Variance）。
    在温柔强度下，FLUX 强大的流匹配流形会"吃下"非正交噪声，Mean Cosine 看起来差不多（甚至 w/o Orthogonal 可能略高），FID 差距也不大。
    但看 Std（标准差）：
    - Full Method 的 Std ≈ 0.01~0.02（全图能量绝对均匀）
    - w/o Orthogonal 的 Std ≈ 0.05~0.10（各 Token 能量像过山车）
    这导致 w/o Orthogonal 无法设定统一阈值做盲提取，差分热力图也无法稳定。

    【极端 alpha 压力测试（α=2.0, 3.0, 4.0）—— 核心卖点】
    在学术界这叫 Crash Test / Stress Test。
    跑一组 Full Method (α=3.0)：严格正交，能量藏在零空间，无论 α 多大，图像依然完美，FID 仍保持 206。
    跑一组 w/o Orthogonal (α=3.0)：没有正交保护，巨大噪声直接和语义撞车，猫会变成长满彩色斑块的扭曲怪物，FID 暴涨到 350+！
    论文话术： "At standard injection strengths, FLUX's robust manifold conceals the semantic interference of non-orthogonal noise. However, our stress test (α=3.0) reveals the true power of OrthoFlow: while naive injection causes catastrophic semantic collapse (FID surges to 350), orthogonal drift mathematically decouples the watermark from the image semantics, maintaining pristine visual fidelity regardless of injection strength."
    （一句话：正交机制让我们实现了"水印强度与画质的彻底解耦"！）

  c）结合语义去做的后果
    我们已改为全图 Global 注入，不再依赖语义 Mask。原因在于：正交投影已经保证了任何区域注入都不会毁画质。若仍使用语义 Mask，背景无水印会导致角落篡改完全检测不到（0−0=0），系统变瞎子。全图注入是正交流形的自然延伸与必然选择。

  d）网格分辨率消融 (Grid Size Ablation)
    对比 32×32、16×16、8×8 的提取效果。
    预期结论：32×32 虽然理论空间精度最高，但向量仅 64D，余弦相似度统计学方差极大，假阳性飙升，F1-Score 暴跌。16×16（256D）有所改善但仍不够稳定。8×8（1024D）是统计学稳定与空间精度的最优解（Sweet Spot）。
    但注意：我们现在已经改为 32×32 原生分辨率做篡改定位（配合形态学正则化），全局检测仍建议用更大聚合块提升稳定性。

  e）盲提取消融 (Blind Extraction Ablation)
    对比 "有原始 Prompt" vs "空 Prompt (\"\")" 的提取效果。
    预期结论：由于 DiT 全局注意力机制，即使不提供任何文本提示，空 Prompt 提取的 Mean Cosine 仍与有 Prompt 极其接近（差距 < 5%）。
    这证明 OrthoFlow 实现了 "Prompt-agnostic Blind Extraction"—— 既不需要原始图像，也不需要原始 Prompt，是学术界最高级别的 Fully Blind 验证。

确认水印是否好用？
隐蔽性：正交，欧拉步进加权，隐空间嵌入：计算 clip/fid数值
  【注意】小样本（如 20 张图）下 FID 绝对值可能较高（如 200+），这是正常的。当你开启大规模实验（1000 张或 5000 张）时，FID 会自然降到 30 甚至 20 左右。小批量测试时只看 ΔFID（Full Method vs Clean 的差距），只要差距极小（< 5）即证明水印零语义干扰。
鲁棒性：分块签名设计，Transformer 特征提取，能量重归一化
安全性：密钥唯一性，无密钥不可验证，无密钥不可嵌入：加密

Limitations（坦诚面对，加分项）
- Vulnerability to Spatial Desynchronization（对空间去同步的脆弱性）：中心裁剪/平移攻击会导致检测性能下降。经过 padding 实验验证，这并非 padding 策略问题，而是 VAE Convolutional Pollution 与 DiT Global Attention Shift 共同导致的不可逆污染。这是所有隐空间网格水印的结构性痛点（Achilles' Heel）。


看我们的机制，把代码进行重组，分为
0 config.json
  保存需要读取的配置信息，下面的功能性 python 文件从这个文件查询配置信息
  "num_inference_steps": 4,
  "guidance_scale": 0.0,
  "height": 512,
  "width": 512,
  "alpha": 0.6,
  "fft_radius_inner": 2,
  "fft_radius_outer": 14,
  "secret_key": 42
1 no_watermark_img_generate.py
  用prompt.json里的 prompt
  生成没有水印的图片并保存到/home/daiyn/project_flux/six/pic/no_watermarked_img/
2 watermarked_img_generate.py
  用prompt.json里的 prompt
  生成加了水印的图片并保存到/home/daiyn/project_flux/six/pic/watermarked_img/
3 attack_img_generate.py 在加了水印的图片的基础性加攻击，生成被攻击后的图片
  jpeg压缩（75, 50, 30, 20, 10）
  高斯模糊（0.5, 1.0, 2.0, 3.0, 4.0）
  中心裁剪
  加噪声（0.03, 0.05, 0.10, 0.15, 0.20）
  缩放（0.75, 0.50, 0.25）
  亮度/对比度调整 (Brightness / Contrast)
  图片随机加入黑色方块
  sdxl/flux语义修改：使用prompt_modified.json的 prompt 和 全局风格迁移（比如换成油画风格，换成素描风格）
  SDEdit (扩散再生攻击 / Image-to-Image)： noise_strength = 0.3, 0.4, 0.5

  生成图片并且保存到/home/daiyn/project_flux/six/pic/attack_watermarked_img/jpeg50（如果攻击方式是 jpeg 压缩 50）、生成图片并且保存到/home/daiyn/project_flux/six/pic/attack_watermarked_img/Brightness50（如果攻击方式是调整Brightness为 50% ）以此类推
4 extract_watermark.py
  内含提取水印函数，所有需要提取水印的操作都调用这个文件的函数
  输入：需要提取的图片数量，需要提取水印的图片路径，输出32 * 32 签名 S 的路径
  输出：32 * 32 签名 S
  读取来源需要可以选，可以是无水印的/有水印的/某种消融实验的/某种攻击方式的
  支持两种提取模式：
    - 有 Prompt 模式：使用原始 prompt 编码（默认）
    - 空 Prompt 模式：使用 "" 的 prompt embedding 进行 Fully Blind 提取
  保存到/home/daiyn/project_flux/six/watermark_extra/
  每种来源的输出要单独一个文件夹
5 analyze_robustness_result.py
  读取提取的水印/home/daiyn/project_flux/six/watermark_extra/
  读取来源需要可以选，可以是无水印的/有水印的/某种消融实验的/某种攻击方式的
  根据加水印的原图提取的签名 S_orig，被攻击的图像中提取的签名S_tamp
  计算统计核心区域的均值、标准差、最小值和最大值

  维度一：全局鲁棒性评价 (Robustness / Copyright Verification)
  针对的攻击：JPEG压缩、高斯模糊、加噪声、缩放等（全图破坏，没有局部篡改）。
  核心逻辑：证明攻击后，提取出的签名 $S_{tamp}$ 依然能证明图片是你的。不需要做减法（不看 $S_{orig} - S_{tamp}$），只看 $S_{tamp}$ 的绝对强度。

  你除了算均值（Mean）、标准差（Std）之外，还须计算以下指标：

  1. TPR @ fixed FPR (例如 TPR@0.1% FPR)
    在控制误报率（假阳性率 FPR）极其低的情况下（比如万分之一），你的检出率（真阳性率 TPR）是多少。
    一批没有加过水印的图（比如 100 张），去提取它们的均值，计算出一个“非水印图的得分分布”（通常均值在 0 左右）。然后设定一个阈值 $\tau$ 使得 FPR=0.1%。接着看你的受攻击图像 $S_{tamp}$ 的均值有多少个大于 $\tau$。
  2. Detection AUC-ROC (检测级的 ROC 曲线下面积)
    衡量分类器（有水印 vs 无水印）综合性能的指标。0.5 是瞎猜，1.0 是完美检测。
    把受攻击图的分数作为正样本，无水印图/自然图像的分数作为负样本，调用 `sklearn.metrics.roc_auc_score`。
  3. Retention Rate (信号保留率)
    `Mean(S_tamp) / Mean(S_orig) * 100%`
  4. 【新增】签名内部标准差 Std Cosine
    计算每个签名 S 自身的标准差，然后对所有图像取平均。
    这是正交机制的核心评估指标：Full Method 的 Std 极低（能量均匀），w/o Orthogonal 的 Std 极高（能量过山车）。

  维度二：篡改定位评价 (Tamper Localization / Fragility)
  针对的攻击：随机黑色方块、局部重绘（Inpainting）、局部语义修改。
  核心逻辑：评价你画出的“差分热力图” $\Delta S = |S_{orig} - S_{tamp}|$ 有多精准。
  这是一个图像分割/异常检测（Anomaly Detection）问题。
  1. Pixel/Patch-level AUC-ROC (像素/块级 AUC)
  2. F1-Score / Dice Coefficient
  3. IoU (Intersection over Union, 交并比)
    预测的红方块和真实的黑方块重合的面积占比。
  4. 【必须加 Baseline 对比】在同一表格中加入 Tree-Ring 等前作方法的定位指标。
    Tree-Ring 会被 DiT 全局注意力撕碎，IoU 通常为 0.00~0.05。与你的 0.51 对比，凸显从 0 到 1 的质变。

  Table 1 (Robustness) 的表头应该是：`Attack Type | Mean Cosine | Std Cosine | Retention Rate | AUC | TPR@0.1%FPR`
  Table 2 (Tamper Localization) 的表头应该是：`Attack Type | Patch-AUC | F1-Score | IoU | Baseline IoU`。

  每种来源的输出要单独一个文件夹
  生成图篡改热力图并且保存到/home/daiyn/project_flux/six/result/pic/tamper_img/
  每张图片的分析结果保存到/home/daiyn/project_flux/six/result/detail.json
  对每种攻击的总分析结果保存到/home/daiyn/project_flux/six/result/merge.json

6 analyze_invisibility_result.py
  计算加水印的生成图和不加水印的生成图之间的clip和 fid
  【注意】小样本（20张）下 FID 绝对值高是正常的，只看 ΔFID（与 Clean 的差距）。
  大规模实验（1000+张）后 FID 会自然降到 30 以下。

7.1 ablation_fft.py
 7.2 ablation_obj.py
 7.3 ablation_sem.py
  同上

7.4 ablation_obj_stress.py
  【正交极限压力测试 (Stress Test / Crash Test)】
  对 alpha = 1.0, 2.0, 3.0, 4.0 分别运行：
  - w/o Orthogonal (非正交直接注入)：图像在 alpha≥2.0 时开始扭曲，alpha=3.0 时变成怪物
  - Full Method (正交注入)：即使 alpha=4.0，图像依然完美
  输出目录：
    /home/daiyn/project_flux/six/pic/ablation_obj_stress_watermarked_img/alpha_X.X/
    /home/daiyn/project_flux/six/pic/ablation_ortho_stress_watermarked_img/alpha_X.X/
  每张图计算 32×32 签名，打印每个 alpha 下的 Mean 和 Std 对比。

8 test.sh / test_auto.sh
  把以上所有实验机制的脚本串联，实行。
  注意：test_auto.sh 中 no_watermark + watermarked 使用 fluxenv，后续所有步骤使用 attackenv。

9 prompts.json
  存放从MS-COCO 2017 (Validation Set)和Gustavosta/Stable-Diffusion-Prompts 获取的prompt
10 prompts_modified.json
  存放大模型修改好的 prompt.json的语义修改 prompt（大模型修改这步我自己去做）
  {
    "wm_000": {
      "sdxl_style_oil_painting": {
        "original_prompt": "A frisbee in the air",
        "modified_prompt": "[oil_painting] A frisbee in the air, rendered in rich oil painting style with visible brushstrokes and textured canvas",
        "instruction": "transform into an oil painting style",
        "modification_type": "oil_painting"
      },
      "sdxl_style_sketch": {
        "original_prompt": "A frisbee in the air",
        "modified_prompt": "[sketch] A frisbee in the air, pencil sketch style with cross-hatching, shading and fine line work",
        "instruction": "convert to pencil sketch style",
        "modification_type": "sketch"
      },
      "fluxfill_center": {
        "original_prompt": "A frisbee in the air",
        "modified_prompt": "[inpaint_center] A frisbee in the air (inpainted: a colorful bird perched on the frisbee)",
        "inpaint_prompt": "a colorful bird perched on the frisbee",
        "modification_type": "inpaint_center"
      }
    },

    我需要你的脚本可以让 fluxfill_random复用 fluxfill_center的 prompt，只是把 inpaint_center改成 inpaint_random

11 generate_prompt.py
  从MS-COCO 2017 (Validation Set)和Gustavosta/Stable-Diffusion-Prompts 各随机获取100 条 prompt（随机数种子要固定）（这一步只做一次，之后一直复用生成的 prompt.json）

我的显存大概有 80gb,你需要利用完一个模型之后就卸载，免得 oom
不要用 pipe.enable_vae_tiling()我的依赖包不支持！
我在服务器实验，你把代码写在我的电脑里，我自己去粘贴代码
你可以参考我之前的代码，位置在@2026/code/sixth（本机）
在这个基础上修改代码，如果不符合我们现在的大纲就改掉它！
我已经下载好的模型在/home/daiyn/project_flux/model（服务器）
有/clip-model     /flux-fill     /flux-schnell    /sdxl-instructpix2pix
你的脚本最好是低耦合的，我可能后续需要调用某个方法，我不想重写代码

我有两个虚拟环境一个是用来生成无水印图像和有水印图像的fluxenv
一个是用来生成攻击图像提取水印分析结果的attackenv
他们的依赖包版本已经放在 sixth 这个文件夹里了你可以参考
