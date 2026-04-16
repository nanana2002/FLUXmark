#!/bin/bash
# 安装所有依赖脚本

set -e

echo "=========================================="
echo "  OrthoFlow 依赖安装脚本"
echo "=========================================="
echo ""

# 检测是否安装了 conda
if command -v conda &> /dev/null; then
    echo "✓ 检测到 Conda 环境"
    echo "  当前环境: $CONDA_DEFAULT_ENV"
    read -p "是否继续在当前环境安装? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "请激活正确环境后重试"
        exit 1
    fi
else
    echo "⚠️  未检测到 Conda，使用系统 Python"
fi

echo ""
echo "选择镜像源:"
echo "1) 清华镜像 (推荐，国内 fastest)"
echo "2) 阿里云镜像"
echo "3) 中科大镜像"
echo "4) 官方源 (国外服务器选择)"
read -p "请输入选项 (1-4): " choice

case $choice in
    1)
        PIP_INDEX="https://pypi.tuna.tsinghua.edu.cn/simple"
        echo "使用清华镜像"
        ;;
    2)
        PIP_INDEX="https://mirrors.aliyun.com/pypi/simple"
        echo "使用阿里云镜像"
        ;;
    3)
        PIP_INDEX="https://pypi.mirrors.ustc.edu.cn/simple"
        echo "使用中科大镜像"
        ;;
    4)
        PIP_INDEX="https://pypi.org/simple"
        echo "使用官方源"
        ;;
    *)
        PIP_INDEX="https://pypi.tuna.tsinghua.edu.cn/simple"
        echo "默认使用清华镜像"
        ;;
esac

echo ""
echo "=========================================="
echo "开始安装依赖..."
echo "=========================================="
echo ""

# 升级 pip
python3 -m pip install --upgrade pip -i $PIP_INDEX

# 安装 PyTorch (根据 CUDA 版本选择)
echo ""
echo "检测 CUDA 版本..."
if command -v nvcc &> /dev/null; then
    CUDA_VERSION=$(nvcc --version | grep "release" | sed -n 's/.*release \([0-9]\+\.[0-9]\+\).*/\1/p')
    echo "检测到 CUDA 版本: $CUDA_VERSION"
else
    echo "未检测到 nvcc，使用默认 CUDA 12.1 版本"
    CUDA_VERSION="12.1"
fi

echo ""
read -p "是否安装 GPU 版 PyTorch? (y/n, 默认 y) " -n 1 -r
echo
if [[ $REPLY =~ ^[Nn]$ ]]; then
    echo "安装 CPU 版 PyTorch..."
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu -i $PIP_INDEX
else
    echo "安装 GPU 版 PyTorch (CUDA 12.1)..."
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 -i $PIP_INDEX
fi

# 安装其他依赖
echo ""
echo "安装其他依赖..."
pip install -r requirements.txt -i $PIP_INDEX

echo ""
echo "=========================================="
echo "✓ 依赖安装完成!"
echo "=========================================="
echo ""

# 验证安装
echo "验证关键包安装:"
python3 -c "import torch; print(f'  ✓ PyTorch {torch.__version__}, CUDA可用: {torch.cuda.is_available()}')" || echo "  ✗ PyTorch 导入失败"
python3 -c "import diffusers; print(f'  ✓ diffusers {diffusers.__version__}')" || echo "  ✗ diffusers 导入失败"
python3 -c "import transformers; print(f'  ✓ transformers {transformers.__version__}')" || echo "  ✗ transformers 导入失败"
python3 -c "import modelscope; print(f'  ✓ modelscope 安装成功')" || echo "  ✗ modelscope 导入失败"

echo ""
echo "如果以上包都显示 ✓，则可以开始运行实验:"
echo "  bash test.sh"
echo ""
