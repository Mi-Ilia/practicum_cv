"""Утилиты для визуализации предсказаний."""
from typing import List, Dict, Tuple, Optional
import os
from pathlib import Path
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

try:
    from IPython.display import Image, display
    IPYTHON_AVAILABLE = True
except ImportError:
    IPYTHON_AVAILABLE = False


def compute_iou_dice_per_image(pred_mask: np.ndarray, gt_mask: np.ndarray, num_classes: int = 3, ignore_index: int = 255) -> Dict[str, float]:
    """Вычисляет IoU и Dice для каждого изображения.
    
    Args:
        pred_mask: Предсказанная маска (H, W)
        gt_mask: Ground truth маска (H, W)
        num_classes: Количество классов
        ignore_index: Индекс для игнорирования
        
    Returns:
        Словарь с метриками: {'mIoU': float, 'mDice': float, 'IoU_per_class': list, 'Dice_per_class': list}
    """
    pred_mask = pred_mask.flatten()
    gt_mask = gt_mask.flatten()
    
    # Игнорируем пиксели с ignore_index
    valid_mask = (gt_mask != ignore_index)
    pred_mask = pred_mask[valid_mask]
    gt_mask = gt_mask[valid_mask]
    
    iou_per_class = []
    dice_per_class = []
    
    for cls in range(num_classes):
        pred_cls = (pred_mask == cls)
        gt_cls = (gt_mask == cls)
        
        intersection = np.logical_and(pred_cls, gt_cls).sum()
        union = np.logical_or(pred_cls, gt_cls).sum()
        pred_area = pred_cls.sum()
        gt_area = gt_cls.sum()
        
        # IoU
        if union > 0:
            iou = intersection / union
        else:
            iou = 1.0 if intersection == 0 else 0.0
        iou_per_class.append(iou)
        
        # Dice
        if pred_area + gt_area > 0:
            dice = 2 * intersection / (pred_area + gt_area)
        else:
            dice = 1.0 if intersection == 0 else 0.0
        dice_per_class.append(dice)
    
    # Средние значения (исключаем фон для более информативной метрики)
    # Обрабатываем NaN: если класс отсутствует, IoU/Dice может быть NaN
    iou_per_class_clean = [x for x in iou_per_class[1:] if not np.isnan(x)] if len(iou_per_class) > 1 else []
    dice_per_class_clean = [x for x in dice_per_class[1:] if not np.isnan(x)] if len(dice_per_class) > 1 else []
    
    if len(iou_per_class_clean) > 0:
        mIoU = np.mean(iou_per_class_clean)
    elif len(iou_per_class) > 1:
        mIoU = 0.0  # Если все классы дали NaN, ставим 0
    else:
        mIoU = iou_per_class[0] if not np.isnan(iou_per_class[0]) else 0.0
    
    if len(dice_per_class_clean) > 0:
        mDice = np.mean(dice_per_class_clean)
    elif len(dice_per_class) > 1:
        mDice = 0.0  # Если все классы дали NaN, ставим 0
    else:
        mDice = dice_per_class[0] if not np.isnan(dice_per_class[0]) else 0.0
    
    # Заменяем NaN на 0.0 для безопасности
    iou_per_class_safe = [0.0 if np.isnan(x) else float(x) for x in iou_per_class]
    dice_per_class_safe = [0.0 if np.isnan(x) else float(x) for x in dice_per_class]
    
    return {
        'mIoU': float(mIoU),
        'mDice': float(mDice),
        'IoU_per_class': iou_per_class_safe,
        'Dice_per_class': dice_per_class_safe
    }


def get_palette(num_classes: int = 3) -> List[Tuple[int, int, int]]:
    """Возвращает палитру цветов для классов."""
    if num_classes == 3:
        return [(0, 0, 0), (255, 0, 0), (0, 255, 0)]  # Black, Red, Green
    else:
        # Генерируем палитру для большего количества классов
        palette = [(0, 0, 0)]  # Фон - черный
        colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255),
            (255, 255, 0), (255, 0, 255), (0, 255, 255),
            (128, 0, 0), (0, 128, 0), (0, 0, 128),
        ]
        for i in range(1, num_classes):
            palette.append(colors[(i - 1) % len(colors)])
        return palette


def visualize_prediction(image: np.ndarray, pred_mask: np.ndarray, gt_mask: np.ndarray,
                        metrics: Dict[str, float], image_name: str, palette: List[Tuple[int, int, int]],
                        class_names: Optional[List[str]] = None, save_path: Optional[str] = None):
    """Визуализирует одно предсказание в формате 1x4 (одна строка из 4 изображений).
    
    Формат:
    1. Original (чистое фото)
    2. Ground Truth (оригинал + зеленая полупрозрачная маска)
    3. Prediction (оригинал + желтая маска) + метрики в заголовке
    4. Error Map (TP=зеленый, FP=красный, FN=синий)
    
    Args:
        image: Исходное изображение (H, W, 3)
        pred_mask: Предсказанная маска (H, W)
        gt_mask: Ground truth маска (H, W)
        metrics: Словарь с метриками
        image_name: Имя изображения
        palette: Палитра цветов
        class_names: Имена классов
        save_path: Путь для сохранения
    """
    # Проверяем размеры
    if image.shape[:2] != gt_mask.shape[:2] or image.shape[:2] != pred_mask.shape[:2]:
        # Приводим к одному размеру (берем минимальный)
        min_h = min(image.shape[0], gt_mask.shape[0], pred_mask.shape[0])
        min_w = min(image.shape[1], gt_mask.shape[1], pred_mask.shape[1])
        image = image[:min_h, :min_w]
        gt_mask = gt_mask[:min_h, :min_w]
        pred_mask = pred_mask[:min_h, :min_w]
    
    # Нормализуем изображение если нужно
    if image.max() > 1.0:
        img_show = image.astype(np.float32) / 255.0
    else:
        img_show = image.astype(np.float32)
    
    # Создаем фигуру с 4 колонками
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    plt.subplots_adjust(wspace=0.1)
    
    # Валидные пиксели (игнорируем ignore_index=255 и значения >= num_classes)
    valid_mask = (gt_mask < len(palette)) & (gt_mask != 255)
    valid_pred_mask = (pred_mask < len(palette)) & (pred_mask != 255)
    
    # Функция-помощник для наложения полупрозрачной маски на изображение
    def apply_mask_overlay(img, mask, color, alpha=0.4):
        """Накладывает полупрозрачную маску цвета на изображение."""
        overlay = np.zeros_like(img)
        mask_bool = mask > 0
        if mask_bool.any():
            overlay[mask_bool] = color
            # Смешиваем: (1-alpha)*img + alpha*overlay
            result = img.copy()
            result[mask_bool] = img[mask_bool] * (1 - alpha) + overlay[mask_bool] * alpha
            return result
        return img
    
    # 1. ORIGINAL (чистое фото)
    axes[0].imshow(img_show)
    axes[0].set_title(f"1. Input\n{image_name}", fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    # 2. GROUND TRUTH (оригинал + зеленая полупрозрачная маска)
    # Создаем маску для всех классов кроме фона (класс 0)
    gt_foreground = (gt_mask > 0) & valid_mask
    # Используем зеленый цвет для GT
    gt_overlay_color = np.array([0.0, 1.0, 0.0])  # Зеленый в RGB [0,1]
    viz_gt = apply_mask_overlay(img_show, gt_foreground, gt_overlay_color, alpha=0.4)
    axes[1].imshow(viz_gt)
    axes[1].set_title("2. Ground Truth", fontsize=12, fontweight='bold')
    axes[1].axis('off')
    
    # 3. PREDICTION (оригинал + желтая/оранжевая маска) + МЕТРИКИ В ЗАГОЛОВКЕ
    # Создаем маску для всех классов кроме фона
    pred_foreground = (pred_mask > 0) & valid_pred_mask
    # Используем желтый/золотой цвет для предсказания
    pred_overlay_color = np.array([1.0, 0.8, 0.0])  # Золотой/желтый в RGB [0,1]
    viz_pred = apply_mask_overlay(img_show, pred_foreground, pred_overlay_color, alpha=0.4)
    axes[2].imshow(viz_pred)
    
    # Метрики в заголовке (Вариант А - рекомендованный)
    iou_score = metrics.get('mIoU', 0.0)
    dice_score = metrics.get('mDice', 0.0)
    title_text = f"3. Prediction\nIoU: {iou_score:.3f} | Dice: {dice_score:.3f}"
    
    # Цвет заголовка зависит от качества (зеленый если хорошо, красный если плохо)
    title_color = 'green' if iou_score > 0.7 else 'orange' if iou_score > 0.5 else 'red'
    
    axes[2].set_title(title_text, fontsize=12, fontweight='bold', color=title_color)
    axes[2].axis('off')
    
    # 4. ERROR MAP (TP=зеленый, FP=красный, FN=синий)
    # Вычисляем компоненты ошибок только для валидных пикселей
    # TP (True Positive): правильно предсказанные пиксели (совпали с GT)
    tp = (pred_mask == gt_mask) & (pred_mask > 0) & valid_mask & valid_pred_mask
    
    # FP (False Positive): лишние пиксели (предсказано, но нет в GT)
    fp = (pred_mask > 0) & (gt_mask == 0) & valid_pred_mask & valid_mask
    
    # FN (False Negative): пропущенные пиксели (есть в GT, но не предсказано)
    fn = (gt_mask > 0) & (pred_mask == 0) & valid_mask & valid_pred_mask
    
    # Создаем карту ошибок: затемняем фон, накладываем цвета
    viz_err = img_show * 0.5  # Затемняем фон на 50%
    
    # TP - зеленый (правильно предсказанные, можно сделать менее ярким)
    if tp.any():
        viz_err[tp] = viz_err[tp] * 0.7 + np.array([0.0, 0.8, 0.0]) * 0.3
    
    # FP - красный (лишнее предсказание)
    if fp.any():
        viz_err[fp] = np.array([1.0, 0.0, 0.0])  # Яркий красный
    
    # FN - синий (пропущенное)
    if fn.any():
        viz_err[fn] = np.array([0.0, 0.5, 1.0])  # Небесно-синий
    
    axes[3].imshow(viz_err)
    axes[3].set_title("4. Errors\nRed: FP (Extra) | Blue: FN (Missed)", fontsize=10)
    axes[3].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def show_images_from_folder(folder_path: str, limit: Optional[int] = None):
    """Отображает изображения из указанной папки в Jupyter.
    
    Args:
        folder_path: Путь к папке с изображениями
        limit: Максимум изображений для отображения (если None, показываем все)
    
    Example:
        >>> from src.utils.visualization_utils import show_images_from_folder
        >>> show_images_from_folder('experiments/h1_experiment/val_results/visualizations/best_predictions', limit=5)
    """
    if not IPYTHON_AVAILABLE:
        print("IPython не доступен. Эта функция работает только в Jupyter notebook.")
        return
    
    if not os.path.exists(folder_path):
        print(f"Папка не найдена: {folder_path}")
        return
    
    # Поддерживаем несколько популярных форматов
    patterns = ["*.png", "*.jpg", "*.jpeg"]
    paths = []
    for pattern in patterns:
        paths.extend(glob(os.path.join(folder_path, pattern)))
    
    paths = sorted(paths)
    if limit is not None:
        paths = paths[:limit]
    
    if not paths:
        print(f"Нет изображений в папке: {folder_path}")
        return
    
    for path in paths:
        print(os.path.basename(path))
        display(Image(filename=path))


def collect_best_worst_predictions(
    experiment_name: str,
    src_root: str = "experiments",
    dst_root: str = "supplementary/viz/predictions",
    prefix: Optional[str] = None,
    limit_best: int = 3,
    limit_worst: int = 3,
    split: str = "val",
):
    """
    Копирует top-N лучших и худших предсказаний из эксперимента в общую папку
    для отчёта с добавлением префикса (например, h1_, h2_, exp1_).

    Args:
        experiment_name: Имя эксперимента (например, "h1_experiment")
        src_root: Корневая директория с экспериментами (по умолчанию "experiments")
        dst_root: Целевая директория для копирования (по умолчанию "supplementary/viz/predictions")
        prefix: Префикс для имен файлов (например, "h1", "h2", "exp1"). 
                Если None, используется имя эксперимента
        limit_best: Количество лучших предсказаний для копирования (по умолчанию 3)
        limit_worst: Количество худших предсказаний для копирования (по умолчанию 3)
        split: Тип результатов - "val" или "test" (по умолчанию "val")

    Пример:
        >>> from src.utils.visualization_utils import collect_best_worst_predictions
        >>> # Валидационные результаты
        >>> collect_best_worst_predictions("h1_experiment", prefix="h1", split="val")
        >>> collect_best_worst_predictions("h2_experiment", prefix="h2", split="val")
        >>> # Тестовые результаты
        >>> collect_best_worst_predictions("exp1_experiment", prefix="exp1", split="test")
        >>> collect_best_worst_predictions("exp2_experiment", prefix="exp2", split="test")

    Структура ожидается такая:
        experiments/<experiment_name>/val_results/visualizations/best_predictions/*.png
        experiments/<experiment_name>/val_results/visualizations/worst_predictions/*.png
        или
        experiments/<experiment_name>/test_results/visualizations/best_predictions/*.png
        experiments/<experiment_name>/test_results/visualizations/worst_predictions/*.png
    """
    import shutil
    
    experiment_name = experiment_name.rstrip("/\\")
    if prefix is None:
        prefix = experiment_name  # если явно не указали, используем имя эксперимента

    # Определяем путь к результатам в зависимости от split
    if split == "test":
        results_dir = "test_results"
    else:
        results_dir = "val_results"
    
    src_base = Path(src_root) / experiment_name / results_dir / "visualizations"
    best_dir = src_base / "best_predictions"
    worst_dir = src_base / "worst_predictions"

    dst_dir = Path(dst_root)
    dst_dir.mkdir(parents=True, exist_ok=True)

    if not best_dir.exists():
        print(f"[WARN] Нет папки с лучшими предсказаниями: {best_dir}")
    if not worst_dir.exists():
        print(f"[WARN] Нет папки с худшими предсказаниями: {worst_dir}")

    def copy_group(src_dir: Path, tag: str, limit: int):
        """Копирует группу файлов с префиксом."""
        if not src_dir.exists():
            return
        
        files = sorted(
            [p for p in src_dir.iterdir() if p.is_file() and p.suffix.lower() in [".png", ".jpg", ".jpeg"]]
        )
        if limit is not None:
            files = files[:limit]

        copied_count = 0
        for p in files:
            # Добавляем split в имя файла для различия val/test
            new_name = f"{prefix}_{split}_{tag}_{p.name}"
            dst_path = dst_dir / new_name
            shutil.copy2(p, dst_path)
            copied_count += 1
        
        if copied_count > 0:
            print(f"  Copied {copied_count} {tag} prediction(s) from {split}")

    # Копируем лучшие и худшие
    print(f"Collecting {split} predictions from {experiment_name}...")
    copy_group(best_dir, "best", limit_best)
    copy_group(worst_dir, "worst", limit_worst)
    print(f"Done! Results saved to: {dst_dir}")
