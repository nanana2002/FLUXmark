import sys

print(f"Python: {sys.version}")
print(f"Path: {sys.executable}")
print("-" * 50)

for m in ['torch', 'diffusers', 'transformers', 'accelerate', 'torchvision', 'numpy', 'PIL']:
    try:
        x = __import__(m)
        if m == 'torch':
            print(f"torch: {x.__version__}")
            print(f"  CUDA available: {x.cuda.is_available()}")
            if x.cuda.is_available():
                print(f"  CUDA version: {x.version.cuda}")
                print(f"  GPU: {x.cuda.get_device_name(0)}")
        else:
            print(f"{m}: {x.__version__}")
    except Exception as e:
        print(f"{m}: ERROR ({e})")

print("-" * 50)
