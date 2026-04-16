#!/usr/bin/env python3
"""
模型下载脚本 V3 - 修复 snapshot_download 参数问题
使用 cache_dir + 手动复制到目标位置
"""

from modelscope import snapshot_download
import os
import sys
import time
import json
import shutil
from datetime import datetime

# 模型根目录
MODEL_ROOT = '/data/daiyina/project_flux/model'
os.makedirs(MODEL_ROOT, exist_ok=True)

# 日志目录
LOG_DIR = '/data/daiyina/project_flux/log'
os.makedirs(LOG_DIR, exist_ok=True)

LOG_FILE = os.path.join(LOG_DIR, 'model_download.log')
PROGRESS_FILE = os.path.join(LOG_DIR, 'model_download_progress.json')

def log(msg):
    """输出日志"""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_msg = f"[{timestamp}] {msg}"
    print(log_msg)
    with open(LOG_FILE, 'a') as f:
        f.write(log_msg + '\n')
        f.flush()

def save_progress(model_name, status, path=None, error=None):
    """保存下载进度"""
    progress = {}
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, 'r') as f:
            progress = json.load(f)
    
    progress[model_name] = {
        'status': status,
        'path': path,
        'error': error,
        'timestamp': datetime.now().isoformat()
    }
    
    with open(PROGRESS_FILE, 'w') as f:
        json.dump(progress, f, indent=2)

def load_progress():
    """加载下载进度"""
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, 'r') as f:
            return json.load(f)
    return {}

def get_dir_size(path):
    """计算目录大小（GB）"""
    total = 0
    if os.path.exists(path):
        for dirpath, dirnames, filenames in os.walk(path):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                if os.path.exists(fp):
                    total += os.path.getsize(fp)
    return total / (1024**3)

def copy_tree(src, dst):
    """复制整个目录树"""
    if os.path.exists(dst):
        shutil.rmtree(dst)
    shutil.copytree(src, dst)

def download_with_cache(model_id, local_dir, allow_patterns=None, ignore_patterns=None):
    """
    使用 cache_dir 下载，然后复制到目标位置
    ModelScope 的 snapshot_download 会自动管理缓存
    """
    cache_root = os.path.expanduser('~/.cache/modelscope')
    
    log(f"   开始下载: {model_id}")
    log(f"   临时缓存: {cache_root}")
    log(f"   最终目标: {local_dir}")
    
    # 下载到缓存
    cache_dir = snapshot_download(
        model_id,
        allow_patterns=allow_patterns,
        ignore_patterns=ignore_patterns,
    )
    
    log(f"   下载完成，缓存位置: {cache_dir}")
    
    # 复制到目标位置
    if os.path.exists(local_dir):
        log(f"   清理旧目录: {local_dir}")
        shutil.rmtree(local_dir)
    
    log(f"   复制到目标位置...")
    copy_tree(cache_dir, local_dir)
    log(f"   复制完成")
    
    return local_dir

def download_sdxl_instructpix2pix():
    """下载 SDXL InstructPix2Pix"""
    model_name = "sdxl-instructpix2pix"
    model_id = "AI-ModelScope/instruct-pix2pix"
    local_dir = os.path.join(MODEL_ROOT, "sdxl-instructpix2pix")
    
    log("="*60)
    log(f"📥 开始下载: {model_name}")
    log(f"   目标路径: {local_dir}")
    log("="*60)
    
    save_progress(model_name, 'downloading')
    
    try:
        # 检查是否已存在且完整
        if os.path.exists(local_dir):
            size = get_dir_size(local_dir)
            if size > 5:  # 大于5GB认为已下载完成
                log(f"   发现完整下载 ({size:.2f} GB)，跳过")
                save_progress(model_name, 'completed', local_dir)
                return local_dir
            else:
                log(f"   发现不完整下载 ({size:.2f} GB)，重新下载")
        
        # fp16最小版配置
        allow_patterns = [
            "*.json",
            "*.fp16.safetensors",
            "fp16/*",
            "text_encoder/*",
            "vae/*",
            "unet/*",
            "tokenizer/*",
            "scheduler/*",
            "model_index.json",
        ]
        
        ignore_patterns = [
            "*.bin", "*.pt", "*.ckpt",
            "optimizer*", "checkpoint*",
            "*.msgpack", "*.h5",
        ]
        
        log("   配置: fp16最小版 (~6GB)")
        log("   开始下载... (预计10-30分钟)")
        
        start_time = time.time()
        
        download_with_cache(
            model_id,
            local_dir,
            allow_patterns=allow_patterns,
            ignore_patterns=ignore_patterns,
        )
        
        elapsed = time.time() - start_time
        size_gb = get_dir_size(local_dir)
        
        log(f"✅ 下载完成!")
        log(f"   路径: {local_dir}")
        log(f"   大小: {size_gb:.2f} GB")
        log(f"   耗时: {elapsed/60:.1f} 分钟")
        
        save_progress(model_name, 'completed', local_dir)
        return local_dir
        
    except Exception as e:
        error_msg = str(e)
        log(f"❌ 下载失败: {error_msg}")
        save_progress(model_name, 'failed', error=error_msg)
        raise

def download_flux_fill():
    """下载 FLUX.1-Fill-dev"""
    model_name = "flux-fill"
    model_id = "AI-ModelScope/FLUX.1-Fill-dev"
    local_dir = os.path.join(MODEL_ROOT, "flux-fill")
    
    log("="*60)
    log(f"📥 开始下载: {model_name}")
    log(f"   目标路径: {local_dir}")
    log("="*60)
    
    save_progress(model_name, 'downloading')
    
    try:
        # 检查是否已存在
        if os.path.exists(local_dir):
            size = get_dir_size(local_dir)
            if size > 12:  # 大于12GB认为已下载完成
                log(f"   发现完整下载 ({size:.2f} GB)，跳过")
                save_progress(model_name, 'completed', local_dir)
                return local_dir
            else:
                log(f"   发现不完整下载 ({size:.2f} GB)，重新下载")
        
        allow_patterns = [
            "*.json",
            "*.safetensors",
            "transformer/*",
            "text_encoder/*",
            "text_encoder_2/*",
            "vae/*",
            "scheduler/*",
            "tokenizer/*",
            "tokenizer_2/*",
            "model_index.json",
        ]
        
        ignore_patterns = ["*.pt", "*.ckpt", "optimizer*", "checkpoint*"]
        
        log("   配置: bf16最小版 (~15GB)")
        log("   开始下载... (预计30-60分钟)")
        
        start_time = time.time()
        
        download_with_cache(
            model_id,
            local_dir,
            allow_patterns=allow_patterns,
            ignore_patterns=ignore_patterns,
        )
        
        elapsed = time.time() - start_time
        size_gb = get_dir_size(local_dir)
        
        log(f"✅ 下载完成!")
        log(f"   路径: {local_dir}")
        log(f"   大小: {size_gb:.2f} GB")
        log(f"   耗时: {elapsed/60:.1f} 分钟")
        
        save_progress(model_name, 'completed', local_dir)
        return local_dir
        
    except Exception as e:
        error_msg = str(e)
        log(f"⚠️ 主镜像失败: {error_msg}")
        log("   尝试备用镜像...")
        
        # 备用：fp8量化版
        try:
            model_id = "Kijai/flux-fp8"
            log(f"   尝试: {model_id}")
            
            start_time = time.time()
            
            download_with_cache(model_id, local_dir)
            
            elapsed = time.time() - start_time
            size_gb = get_dir_size(local_dir)
            
            log(f"✅ 备用镜像下载完成!")
            log(f"   路径: {local_dir}")
            log(f"   大小: {size_gb:.2f} GB")
            log(f"   耗时: {elapsed/60:.1f} 分钟")
            
            save_progress(model_name, 'completed', local_dir)
            return local_dir
            
        except Exception as e2:
            error_msg = f"主: {error_msg}, 备: {str(e2)}"
            log(f"❌ 都失败: {error_msg}")
            save_progress(model_name, 'failed', error=error_msg)
            raise

def check_disk_space():
    """检查磁盘空间"""
    stat = os.statvfs(MODEL_ROOT)
    available_gb = (stat.f_bavail * stat.f_frsize) / (1024**3)
    required_gb = 25
    
    log(f"💾 磁盘空间检查:")
    log(f"   可用: {available_gb:.2f} GB")
    log(f"   需要: {required_gb:.2f} GB")
    
    if available_gb < required_gb:
        log(f"❌ 空间不足！还差 {required_gb - available_gb:.2f} GB")
        return False
    return True

def main():
    log("="*60)
    log("🚀 模型下载脚本 V3")
    log(f"   模型目录: {MODEL_ROOT}")
    log(f"   日志文件: {LOG_FILE}")
    log("="*60)
    
    # 检查空间
    if not check_disk_space():
        sys.exit(1)
    
    # 检查已有进度
    progress = load_progress()
    existing = {}
    for name, info in progress.items():
        if info.get('status') == 'completed':
            path = info.get('path', '')
            if os.path.exists(path):
                size = get_dir_size(path)
                if (name == 'sdxl-instructpix2pix' and size > 5) or \
                   (name == 'flux-fill' and size > 12):
                    existing[name] = path
                    log(f"⏩ {name} 已存在 ({size:.2f} GB): {path}")
    
    results = {}
    
    # 下载 SDXL
    if 'sdxl-instructpix2pix' not in existing:
        try:
            path = download_sdxl_instructpix2pix()
            results['sdxl-instructpix2pix'] = path
        except Exception as e:
            log(f"⚠️ SDXL失败: {e}")
    else:
        results['sdxl-instructpix2pix'] = existing['sdxl-instructpix2pix']
    
    # 下载 FLUX Fill
    if 'flux-fill' not in existing:
        try:
            path = download_flux_fill()
            results['flux-fill'] = path
        except Exception as e:
            log(f"⚠️ FLUX失败: {e}")
    else:
        results['flux-fill'] = existing['flux-fill']
    
    # 最终报告
    log("="*60)
    log("📊 下载完成报告")
    log("="*60)
    
    for name, path in results.items():
        if path and os.path.exists(path):
            size = get_dir_size(path)
            log(f"✅ {name}:")
            log(f"   路径: {path}")
            log(f"   大小: {size:.2f} GB")
    
    # 总大小
    total = sum(get_dir_size(p) for p in results.values() if p)
    log(f"\n💾 总1111占用: {total:.2f} GB")
    
    # 使用示例
    log("\n" + "="*60)
    log("📖 使用示例")
    log("="*60)
    
    if 'sdxl-instructpix2pix' in results:
        log("""
SDXL InstructPix2Pix:
    from diffusers import StableDiffusionInstructPix2PixPipeline
    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        "/data/daiyina/project_flux/model/sdxl-instructpix2pix",
        torch_dtype=torch.float16,
    )
""")
    
    if 'flux-fill' in results:
        log("""
FLUX Fill:
    from diffusers import FluxFillPipeline
    pipe = FluxFillPipeline.from_pretrained(
        "/data/daiyina/project_flux/model/flux-fill",
        torch_dtype=torch.bfloat16,
    )
""")
    
    log("="*60)
    log("🎉 完成！")
    log("="*60)

if __name__ == '__main__':
    main()
