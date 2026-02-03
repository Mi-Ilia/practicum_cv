# Инструкция по установке MMDetection на Windows (GPU)

**Конфигурация:** Python 3.10 | PyTorch 2.1.2 | CUDA 12.1 | MMCV 2.1.0

## Предварительные требования
*   **Visual Studio Build Tools (C++):** Необходим для компиляции. Убедитесь, что установлен компонент "Desktop development with C++".
*   **Git:** Установлен и доступен в терминале.
*   **Интернет:** Для скачивания весов моделей.

---

## 1. Подготовка окружения (Важно!)

Версии Python 3.11 и 3.12 **не поддерживаются** для этой сборки PyTorch на Windows. Обязательно используйте Python 3.10.

### Вариант А: Conda
```powershell
conda create -n mmdet_env python=3.10 -y
conda activate mmdet_env
```

### Вариант Б: Стандартный venv
```powershell
# Убедитесь, что у вас установлен Python 3.10
py -3.10 -m venv .env
.\.env\Scripts\activate
```

---

## 2. Установка PyTorch

Ставим стабильную версию 2.1.2 с поддержкой CUDA 12.1. Это критично для работы с GPU.

```powershell
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121
```

**Проверка успеха:**
```powershell
python -c "import torch; print(torch.cuda.is_available())"
```
Должно вывести: `True`

---

## 3. Установка MMCV (Pre-compiled с CUDA)

Используем `openmim` для автоматического скачивания **прекомпилированных колес** с поддержкой CUDA. Это избавит вас от ошибок компиляции C++.

```powershell
pip install -U openmim
mim install "mmcv==2.1.0"
```

**Важно:** Мы фиксируем версию 2.1.0, так как она идеально совместима с PyTorch 2.1.2.

---

## 4. Фикс совместимости NumPy

Современный NumPy 2.x вызывает сбои при импорте PyTorch (`_ARRAY_API not found`). Принудительно понижаем версию:

```powershell
pip install "numpy<2.0.0"
```

---

## 5. Клонирование и установка MMDetection

Клонируем репозиторий и устанавливаем в режиме редактирования (editable), чтобы можно было менять конфиги и код без переустановки.

```powershell
git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection
pip install -v -e .
```

**Ожидаемый результат:** `Successfully installed mmdet-3.3.0`

---

# Проверка установки и Тестирование

## Этап 1: Проверка импорта (Быстро)

Создайте файл `check_install.py` и запустите его:

```python
import torch
import mmcv
import mmdet
from mmcv.ops import RoIAlign, nms

print("="*50)
print(f"PyTorch Version:  {torch.__version__}")
print(f"CUDA Available:   {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU:              {torch.cuda.get_device_name(0)}")

print(f"MMCV Version:     {mmcv.__version__}")
print(f"MMDet Version:    {mmdet.__version__}")
print("-" * 50)

try:
    # Тест CUDA операций
    print("✅ MMCV CUDA Ops loaded successfully")
except ImportError as e:
    print(f"❌ CUDA Ops Failed: {e}")

try:
    # Тест импорта реестра моделей
    from mmdet.registry import MODELS
    print("✅ MMDetection Registry working")
except Exception as e:
    print(f"❌ MMDetection Init Failed: {e}")

print("="*50)
```

**Запуск:**
```powershell
python check_install.py
```

**Ожидаемый результат:**
```
✅ MMCV CUDA Ops loaded successfully
✅ MMDetection Registry working
```

---

## Этап 2: Тестирование на реальной модели (Надежный метод)

Запустите демонстрацию на встроенном изображении с моделью **Faster R-CNN** (эта модель 100% доступна).

**1. Перейдите в папку mmdetection:**
```powershell
cd C:\practicum_cv\sprint_1\minecraft_fcos_yolo\mmdetection
```

**2. Скачайте конфиг и веса модели:**
```powershell
mim download mmdet --config faster-rcnn_r50_fpn_1x_coco --dest .
```
*Это скачает конфиг и файл весов (~170 MB) прямо в текущую папку.*

**3. Запустите демонстрацию инференса:**
```powershell
python demo/image_demo.py demo/demo.jpg faster-rcnn_r50_fpn_1x_coco.py --weights faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth --device cuda:0 --show
```

**Ожидаемый результат:**
- В консоли должны появиться логи обработки.
- **Откроется окно с изображением**, на котором выделены обнаруженные объекты (люди, машины и т.д.) красными или синими рамками.
- В консоли появится: `show the predictions of the demo image`

Если окно открылось и объекты обведены — **установка завершена успешно!**

---

## Решение типичных проблем

| Проблема | Причина | Решение |
| :--- | :--- | :--- |
| `ImportError: numpy.core.multiarray failed to import` | Установлен NumPy 2.0+ | `pip install "numpy<2.0.0"` |
| `ModuleNotFoundError: No module named 'mmcv._ext'` | MMCV без CUDA | Удалите: `pip uninstall mmcv`. Переустановите: `mim install mmcv==2.1.0` |
| `error: Microsoft Visual C++ 14.0 or greater is required` | Нет компилятора C++ | Установить **VS Build Tools** с компонентом "Desktop development with C++" |
| `AssertionError: MMCV==... is incompatible` | Несовпадение версий | Убедитесь, что версии совпадают: `pip show mmcv mmengine torch` |
| `FileNotFoundError: ...pth can not be found` | Файл весов не скачан | Выполните: `mim download mmdet --config faster-rcnn_r50_fpn_1x_coco --dest .` |
| `CUDA out of memory` | Видеопамять переполнена | Уменьшите размер батча в конфиге или используйте модель поменьше |

