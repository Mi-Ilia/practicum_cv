import torch
import mmcv
from mmdet.apis import init_detector
import mmdet

print("="*50)
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available:  {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU Name:        {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version:    {torch.version.cuda}")

print("-" * 20)
print(f"MMCV Version:    {mmcv.__version__}")
print(f"MMDet Version:   {mmdet.__version__}")
print("-" * 20)

# 1. Проверка CUDA-ядер (самое важное)
print("Checking MMCV CUDA Ops...")
try:
    from mmcv.ops import RoIAlign, nms
    print("✅ MMCV CUDA Ops loaded successfully!")
except ImportError as e:
    print(f"❌ ERROR: CUDA Ops not found. {e}")
    print("   Решение: переустановите mmcv с поддержкой CUDA.")

# 2. Проверка инициализации модели (Dummy check)
# Мы просто пробуем импортировать базовые классы, чтобы убедиться, что mmdet не падает
print("Checking MMDetection initialization...")
try:
    # Просто проверка импорта registry, чтобы убедиться, что mmdet встал
    from mmdet.registry import MODELS
    print("✅ MMDetection registry loaded successfully!")
except Exception as e:
    print(f"❌ ERROR: MMDetection init failed. {e}")

print("="*50)
