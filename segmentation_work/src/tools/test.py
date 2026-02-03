import os
import sys
import argparse
from pathlib import Path
import warnings
import logging
import json
from datetime import datetime
from typing import List, Optional, Dict, Any
import time
import numpy as np
import torch
from PIL import Image
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

# Отключаем буферизацию для корректного вывода в Jupyter
if 'PYTHONUNBUFFERED' not in os.environ:
    os.environ['PYTHONUNBUFFERED'] = '1'

warnings.filterwarnings('ignore', category=UserWarning)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
mmseg_path = os.path.join(project_root, 'mmsegmentation')
if os.path.exists(mmseg_path) and mmseg_path not in sys.path:
    sys.path.insert(0, mmseg_path)
sys.path.append(project_root)

from mmengine.config import Config, DictAction
from mmengine.runner import Runner
from mmengine.hooks import Hook
from mmseg.utils import register_all_modules
from src.utils.test_utils import setup_test_config, get_class_names, get_class_names_from_dataloader
from src.utils.visualization_utils import get_palette, visualize_prediction

# Глобальное хранилище для результатов (используется для визуализации)
_collected_results = []

class PerImageMetricsHook(Hook):
    """Хук для сбора результатов и вычисления метрик покартинно во время test().
    
    Для каждой картинки:
    1. Собирает предсказание и GT
    2. Вычисляет метрики используя стандартный IoUMetric (compute_metrics_per_image)
    3. Сохраняет результаты для последующей визуализации
    
    В конце можно усреднить метрики по всем картинкам для получения глобальных метрик.
    """
    
    def __init__(self, num_classes: int = 3):
        super().__init__()
        self.results = []
        self.metrics = []
        self.num_classes = num_classes
        self._first_batch_logged = False
    
    def after_test_iter(self, runner, batch_idx: int, data_batch: dict = None, outputs: list = None):
        """Вычисляет метрики для каждой картинки после каждой итерации."""
        if outputs is None or not outputs:
            return
        
        # Получаем data_samples из data_batch
        data_samples = data_batch.get('data_samples', []) if data_batch else []
        
        for idx, output in enumerate(outputs):
            try:
                # Получаем предсказание
                if not hasattr(output, 'pred_sem_seg') or output.pred_sem_seg is None:
                    continue
                
                pred_data = output.pred_sem_seg.data
                if isinstance(pred_data, torch.Tensor):
                    pred_sem_seg = pred_data.cpu().numpy()
                else:
                    pred_sem_seg = pred_data
                if pred_sem_seg.ndim == 3 and pred_sem_seg.shape[0] == 1:
                    pred_sem_seg = pred_sem_seg[0]
                
                # Получаем GT (из output или из data_samples)
                gt_sem_seg = None
                if hasattr(output, 'gt_sem_seg') and output.gt_sem_seg is not None:
                    gt_data = output.gt_sem_seg.data
                    if isinstance(gt_data, torch.Tensor):
                        gt_sem_seg = gt_data.cpu().numpy()
                    else:
                        gt_sem_seg = gt_data
                    if gt_sem_seg.ndim == 3 and gt_sem_seg.shape[0] == 1:
                        gt_sem_seg = gt_sem_seg[0]
                elif idx < len(data_samples) and hasattr(data_samples[idx], 'gt_sem_seg'):
                    gt_data = data_samples[idx].gt_sem_seg.data
                    if isinstance(gt_data, torch.Tensor):
                        gt_sem_seg = gt_data.cpu().numpy()
                    else:
                        gt_sem_seg = gt_data
                    if gt_sem_seg.ndim == 3 and gt_sem_seg.shape[0] == 1:
                        gt_sem_seg = gt_sem_seg[0]
                
                if gt_sem_seg is None:
                    continue
                
                # Вычисляем метрики для этой картинки используя стандартный IoUMetric
                metrics_dict = compute_metrics_per_image(
                    pred_sem_seg, gt_sem_seg, self.num_classes,
                    ignore_index=255, iou_metrics=['mIoU', 'mDice']
                )
                
                # Получаем путь к изображению
                img_path = None
                if hasattr(output, 'img_path'):
                    img_path = output.img_path
                elif hasattr(output, 'metainfo') and 'img_path' in output.metainfo:
                    img_path = output.metainfo['img_path']
                elif idx < len(data_samples) and hasattr(data_samples[idx], 'img_path'):
                    img_path = data_samples[idx].img_path
                
                # Сохраняем результат
                result = {
                    'output': output,
                    'pred_sem_seg': pred_sem_seg,
                    'gt_sem_seg': gt_sem_seg,
                    'metrics': metrics_dict['main'],  # Средние метрики
                    'per_class_metrics': metrics_dict['per_class'],  # Метрики по классам
                    'img_path': img_path,
                    'batch_idx': batch_idx,
                    'idx': idx
                }
                
                self.results.append(result)
                
            except Exception as e:
                logger.debug(f"Error processing image in batch {batch_idx}, idx {idx}: {e}")
                continue
    
    def after_test_epoch(self, runner, metrics=None):
        """После завершения тестирования сохраняем результаты в глобальное хранилище."""
        global _collected_results
        _collected_results = self.results.copy()
        if len(_collected_results) == 0:
            logger.error(f"❌ PerImageMetricsHook: Collected 0 results!")

def compute_metrics_per_image(pred_mask: np.ndarray, gt_mask: np.ndarray, 
                              num_classes: int, ignore_index: int = 255,
                              iou_metrics: List[str] = ['mIoU', 'mDice']) -> Dict[str, Any]:
    """Вычисляет метрики для одной картинки используя стандартный IoUMetric.
    
    Args:
        pred_mask: Предсказанная маска (H, W) как numpy array
        gt_mask: Ground truth маска (H, W) как numpy array
        num_classes: Количество классов
        ignore_index: Индекс для игнорирования (по умолчанию 255)
        iou_metrics: Список метрик для вычисления ['mIoU', 'mDice']
    
    Returns:
        Словарь с метриками:
        - 'main': средние метрики (mIoU, mDice, mAcc, aAcc)
        - 'per_class': метрики по классам {class_idx: {'IoU': ..., 'Dice': ..., 'Acc': ...}}
    """
    from mmseg.evaluation.metrics.iou_metric import IoUMetric
    
    # Создаем временный IoUMetric
    iou_metric = IoUMetric(
        ignore_index=ignore_index,
        iou_metrics=iou_metrics,
        format_only=False
    )
    
    # Устанавливаем dataset_meta (нужно для работы)
    iou_metric.dataset_meta = {
        'classes': [f'class_{i}' for i in range(num_classes)],
        'label_map': {},
        'reduce_zero_label': False
    }
    
    # Конвертируем маски в torch.Tensor
    pred_tensor = torch.from_numpy(pred_mask.astype(np.int64))
    gt_tensor = torch.from_numpy(gt_mask.astype(np.int64))
    
    # Вычисляем intersect_and_union для одной картинки
    area_intersect, area_union, area_pred_label, area_label = iou_metric.intersect_and_union(
        pred_tensor, gt_tensor, num_classes, ignore_index
    )
    
    # Конвертируем в numpy (безопасная конвертация с учетом разных типов)
    def safe_to_numpy(x):
        if isinstance(x, torch.Tensor):
            return x.cpu().numpy()
        elif isinstance(x, np.ndarray):
            return x
        elif isinstance(x, (np.number, np.bool_)):
            # numpy скаляры - оборачиваем в массив
            return np.asarray(x)
        else:
            return np.array(x)
    
    area_intersect = safe_to_numpy(area_intersect)
    area_union = safe_to_numpy(area_union)
    area_pred_label = safe_to_numpy(area_pred_label)
    area_label = safe_to_numpy(area_label)
    
    # Вычисляем метрики вручную, т.к. total_area_to_metrics имеет баг с numpy скалярами
    ret_metrics = {}
    
    # IoU
    with np.errstate(divide='ignore', invalid='ignore'):
        iou = area_intersect / area_union
        ret_metrics['IoU'] = iou
        
        # Acc (per-class accuracy)
        acc = area_intersect / area_label
        ret_metrics['Acc'] = acc
        
        # Dice
        dice = 2 * area_intersect / (area_pred_label + area_label)
        ret_metrics['Dice'] = dice
    
    # Применяем nan_to_num если нужно
    if iou_metric.nan_to_num is not None:
        ret_metrics = {
            metric: np.nan_to_num(metric_value, nan=iou_metric.nan_to_num)
            for metric, metric_value in ret_metrics.items()
        }
    
    # Вычисляем средние метрики
    ret_metrics_summary = {
        ret_metric: np.round(np.nanmean(ret_metric_value) * 100, 2)
        for ret_metric, ret_metric_value in ret_metrics.items()
    }
    
    # Форматируем средние метрики
    main_metrics = {}
    for key, val in ret_metrics_summary.items():
        if key == 'aAcc':
            main_metrics[key] = float(val)
        else:
            main_metrics['m' + key] = float(val)
    
    # Вычисляем метрики по классам
    per_class_metrics = {}
    if 'IoU' in ret_metrics:
        iou_per_class = ret_metrics['IoU']
        dice_per_class = ret_metrics.get('Dice', np.zeros_like(iou_per_class))
        acc_per_class = ret_metrics.get('Acc', np.zeros_like(iou_per_class))
        
        for class_idx in range(num_classes):
            per_class_metrics[class_idx] = {
                'IoU': float(np.round(iou_per_class[class_idx] * 100, 2)),
                'Dice': float(np.round(dice_per_class[class_idx] * 100, 2)),
                'Acc': float(np.round(acc_per_class[class_idx] * 100, 2))
            }
    
    return {
        'main': main_metrics,
        'per_class': per_class_metrics
    }

# Настройка логирования с явным выводом в stdout для Jupyter
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)  # Явно указываем stdout
    ],
    force=True  # Переопределяем существующую конфигурацию
)
logger = logging.getLogger('TestScript')
# Убеждаемся, что вывод не буферизуется
sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None

def trigger_visualization_hook(cfg, args):
    default_hooks = cfg.default_hooks
    if 'visualization' in default_hooks:
        visualization_hook = default_hooks['visualization']
        # Turn on visualization
        visualization_hook['draw'] = True
        if args.show:
            visualization_hook['show'] = True
            visualization_hook['wait_time'] = args.wait_time
        if args.show_dir:
            visualizer = cfg.visualizer
            visualizer['save_dir'] = args.show_dir
    else:
        raise RuntimeError(
            'VisualizationHook must be included in default_hooks.'
            'refer to usage '
            '"visualization=dict(type=\'VisualizationHook\')"')

    return cfg

def save_metrics_to_json(results, output_dir, config_path, checkpoint_path, split='test', runner=None, cfg=None, 
                        inference_metrics=None, per_class_metrics_from_evaluator=None):
    """Сохраняет метрики в JSON файл.
    
    ⚠️ ВАЖНО: 
    - Глобальные метрики (mIoU, mDice, mAcc, aAcc) берутся из results (evaluator.compute_metrics())
    - Метрики по классам берутся из per_class_metrics_from_evaluator (из evaluator.results)
    - Покартинные метрики НЕ используются для глобальных метрик (только для визуализации)
    """
    if output_dir is None:
        output_dir = os.path.join('./work_dirs', 'test_results')
    os.makedirs(output_dir, exist_ok=True)
    
    classes_info = get_class_names(cfg) if cfg is not None else None
    
    metrics_data = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'config': str(config_path),
            'checkpoint': str(checkpoint_path),
            'split': split,
            'classes': classes_info if classes_info else None
        },
        'metrics': {
            'main': {},
            'per_class': {},
            'inference': {}
        }
    }
    
    all_metrics = {}
    
    if results is not None and isinstance(results, dict):
        all_metrics.update(results)
    
    if runner is not None and hasattr(runner, 'message_hub'):
        try:
            for prefix in ['test', 'val']:
                scalars = runner.message_hub.get_scalar(prefix)
                if scalars:
                    for key, buf in scalars.items():
                        try:
                            all_metrics[key] = buf.current()
                        except Exception:
                            pass
        except Exception:
            pass
    
    # Парсим метрики из results (глобальные метрики из evaluator.compute_metrics())
    # ⚠️ ВАЖНО: Эти метрики правильные (глобальные), не усредненные покартинные!
    per_class_by_idx = {}  # Временное хранилище по индексам
    
    for key, value in all_metrics.items():
        if not isinstance(value, (int, float)):
            continue
            
        value = float(value)
        clean_key = key.replace('test/', '').replace('val/', '')
        
        # Основные метрики (mIoU, mDice, mAcc, aAcc, mFscore, etc.) - ГЛОБАЛЬНЫЕ из evaluator
        if clean_key in ['mIoU', 'mDice', 'mAcc', 'aAcc', 'mFscore', 'mPrecision', 'mRecall']:
            metrics_data['metrics']['main'][clean_key] = value
        # Метрики по классам в формате "IoU_class_0", "Dice_class_1", "Acc_class_2"
        # или "IoU.class_0", "Dice.class_1" (с точкой)
        elif '_class_' in clean_key or '.class_' in clean_key:
            try:
                # Пробуем оба формата: с точкой и с подчеркиванием
                if '.class_' in clean_key:
                    parts = clean_key.split('.class_')
                else:
                    parts = clean_key.split('_class_')
                
                if len(parts) == 2:
                    metric_name = parts[0]  # IoU, Dice, Acc, etc.
                    class_idx = int(parts[1])
                    
                    if class_idx not in per_class_by_idx:
                        per_class_by_idx[class_idx] = {}
                    per_class_by_idx[class_idx][metric_name] = value
            except (ValueError, IndexError):
                metrics_data['metrics']['main'][clean_key] = value
        # Метрики в формате массивов (если приходят как списки)
        elif isinstance(value, (list, tuple)) and len(value) > 0:
            # Если это массив метрик по классам
            if clean_key in ['IoU', 'Dice', 'Acc'] and classes_info:
                for class_idx, metric_value in enumerate(value):
                    if class_idx not in per_class_by_idx:
                        per_class_by_idx[class_idx] = {}
                    per_class_by_idx[class_idx][clean_key] = float(metric_value)
        else:
            # Остальные метрики в main
            metrics_data['metrics']['main'][clean_key] = value
    
    # Добавляем метрики по классам (правильные глобальные метрики)
    # ⚠️ ВАЖНО: Эти метрики вычислены суммированием площадей по всем изображениям
    if per_class_metrics_from_evaluator:
        for class_idx, class_metrics in per_class_metrics_from_evaluator.items():
            if class_idx not in per_class_by_idx:
                per_class_by_idx[class_idx] = {}
            # Приоритет у метрик, вычисленных из покартинных результатов
            per_class_by_idx[class_idx].update(class_metrics)
    
    # Конвертируем per_class_by_idx в формат с именами классов
    if classes_info:
        for class_idx, class_metrics in per_class_by_idx.items():
            if class_idx < len(classes_info):
                class_name = classes_info[class_idx]
                metrics_data['metrics']['per_class'][class_name] = class_metrics
            else:
                # Если индекс выходит за пределы, используем индекс
                metrics_data['metrics']['per_class'][f'Class_{class_idx}'] = class_metrics
    else:
        # Если нет имен классов, используем индексы
        for class_idx, class_metrics in per_class_by_idx.items():
            metrics_data['metrics']['per_class'][f'Class_{class_idx}'] = class_metrics
    
    # Добавляем метрики inference (FPS, latency)
    if inference_metrics:
        metrics_data['metrics']['inference'].update(inference_metrics)
    elif runner is not None:
        # Пытаемся получить информацию о времени выполнения из message_hub
        try:
            if hasattr(runner, 'message_hub'):
                timer_info = runner.message_hub.get_scalar('train') or runner.message_hub.get_scalar('test')
                if timer_info:
                    for key in ['time', 'data_time', 'iter_time']:
                        if key in timer_info:
                            try:
                                time_value = timer_info[key].current()
                                if time_value > 0:
                                    fps = 1000.0 / time_value if 'iter_time' in key else None
                                    latency_ms = time_value
                                    if fps:
                                        metrics_data['metrics']['inference']['fps'] = round(fps, 2)
                                    metrics_data['metrics']['inference']['latency_ms'] = round(latency_ms, 2)
                                    break
                            except Exception:
                                pass
        except Exception:
            pass
    
    # Если inference метрики пустые, удаляем секцию
    if not metrics_data['metrics']['inference']:
        metrics_data['metrics'].pop('inference', None)
    
    metrics_file = os.path.join(output_dir, 'test_metrics.json')
    try:
        with open(metrics_file, 'w', encoding='utf-8') as f:
            json.dump(metrics_data, f, indent=2, ensure_ascii=False)
        logger.info(f"📊 Metrics saved to: {metrics_file}")
        
        if metrics_data['metrics']['main']:
            logger.info("="*80)
            logger.info("📈 Test Metrics Summary:")
            logger.info("-"*80)
            for key, value in metrics_data['metrics']['main'].items():
                logger.info(f"  {key}: {value:.4f}")
            
            if metrics_data['metrics']['per_class']:
                logger.info("-"*80)
                logger.info("📊 Per-Class Metrics:")
                # Сортируем по именам классов или индексам
                sorted_classes = sorted(metrics_data['metrics']['per_class'].keys())
                for class_name in sorted_classes:
                    class_metrics = metrics_data['metrics']['per_class'][class_name]
                    logger.info(f"  {class_name}:")
                    for metric_name, metric_value in class_metrics.items():
                        logger.info(f"    {metric_name}: {metric_value:.4f}")
            
            if 'inference' in metrics_data['metrics'] and metrics_data['metrics']['inference']:
                logger.info("-"*80)
                logger.info("⚡ Inference Metrics:")
                for key, value in metrics_data['metrics']['inference'].items():
                    logger.info(f"  {key}: {value}")
            logger.info("="*80)
    except Exception as e:
        logger.error(f"Failed to save metrics: {e}")


def visualize_top_predictions(runner: Runner, top_n: int, split: str, output_dir: str, class_names: Optional[List[str]] = None):
    """Визуализирует топ N лучших и худших предсказаний используя результаты из первого прохода.
    
    ⚠️ ВАЖНО: Покартинные метрики используются ТОЛЬКО для сортировки top-N.
    Они могут отличаться от глобальных метрик из evaluator (особенно при пустых масках).
    
    Args:
        runner: Runner с загруженной моделью
        top_n: Количество топ предсказаний для визуализации
        split: Раздел датасета (train/val/test) - используется только для получения имен классов
        output_dir: Директория для сохранения визуализаций
        class_names: Имена классов
    """
    global _collected_results
    
    logger.info("="*80)
    logger.info(f"🖼️  Visualizing top {top_n} predictions...")
    
    if not _collected_results or len(_collected_results) == 0:
        logger.warning("⚠️  No results collected from first pass. Visualization skipped.")
        return
    
    
    # Получаем информацию о классах
    model = runner.model
    num_classes = model.decode_head.num_classes if hasattr(model, 'decode_head') else 3
    if class_names is None:
        # Определяем dataloader для получения имен классов
        if split == 'test':
            dataloader = runner.test_dataloader
        elif split == 'val':
            dataloader = runner.val_dataloader
        else:
            dataloader = runner.train_dataloader
        
        if dataloader is not None:
            class_names = get_class_names_from_dataloader(dataloader)
        
        if class_names is None:
            class_names = [f'Class {i}' for i in range(num_classes)]
    
    palette = get_palette(num_classes)
    all_results = []
    
    # Обрабатываем результаты из хука (уже содержат pred_sem_seg, gt_sem_seg, metrics)
    for result_item in _collected_results:
        try:
            pred_sem_seg = result_item['pred_sem_seg'].copy()  # Копируем чтобы не изменять оригинал
            gt_sem_seg = result_item['gt_sem_seg'].copy()
            metrics = result_item['metrics']
            img_path = result_item.get('img_path')
            ori_shape = result_item.get('ori_shape')
            
            # Загружаем изображение
            if img_path:
                try:
                    img = np.array(Image.open(img_path).convert('RGB'))
                    h_img, w_img = img.shape[:2]
                    h_pred, w_pred = pred_sem_seg.shape[:2]
                    
                    # Проверяем соответствие размеров и ресайзим при необходимости
                    if (h_img != h_pred) or (w_img != w_pred):
                        target_h = ori_shape[0] if ori_shape is not None else h_img
                        target_w = ori_shape[1] if ori_shape is not None else w_img
                        
                        # Ресайзим маски к размеру оригинального изображения
                        if CV2_AVAILABLE:
                            pred_sem_seg = cv2.resize(
                                pred_sem_seg.astype(np.uint8),
                                (target_w, target_h),
                                interpolation=cv2.INTER_NEAREST
                            ).astype(pred_sem_seg.dtype)
                            gt_sem_seg = cv2.resize(
                                gt_sem_seg.astype(np.uint8),
                                (target_w, target_h),
                                interpolation=cv2.INTER_NEAREST
                            ).astype(gt_sem_seg.dtype)
                        else:
                            from PIL import Image as PILImage
                            pred_sem_seg = np.array(
                                PILImage.fromarray(pred_sem_seg.astype(np.uint8))
                                .resize((target_w, target_h), PILImage.NEAREST)
                            ).astype(pred_sem_seg.dtype)
                            gt_sem_seg = np.array(
                                PILImage.fromarray(gt_sem_seg.astype(np.uint8))
                                .resize((target_w, target_h), PILImage.NEAREST)
                            ).astype(gt_sem_seg.dtype)
                        
                        # Ресайзим изображение к целевому размеру (если нужно)
                        if (h_img != target_h) or (w_img != target_w):
                            if CV2_AVAILABLE:
                                img = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
                            else:
                                from PIL import Image as PILImage
                                img = np.array(
                                    PILImage.fromarray(img)
                                    .resize((target_w, target_h), PILImage.BILINEAR)
                                )
                except Exception as e:
                    logger.warning(f"Failed to load image {img_path}: {e}")
                    img = np.zeros((pred_sem_seg.shape[0], pred_sem_seg.shape[1], 3), dtype=np.uint8)
            else:
                img = np.zeros((pred_sem_seg.shape[0], pred_sem_seg.shape[1], 3), dtype=np.uint8)
            
            # Получаем имя файла
            if img_path:
                img_name = Path(img_path).stem
            else:
                img_name = f'img_{result_item["batch_idx"]}_{result_item["idx"]}'
            
            all_results.append({
                'image': img,
                'pred_mask': pred_sem_seg,
                'gt_mask': gt_sem_seg,
                'metrics': metrics,
                'image_name': img_name,
                'img_path': img_path
            })
        except Exception as e:
            logger.debug(f"Error processing result item: {e}")
            continue
    
    
    if len(all_results) == 0:
        logger.warning("No results collected, skipping visualization")
        return
    
    # Сортируем по mDice (обрабатываем NaN - ставим их в конец)
    def sort_key(x):
        dice = x['metrics']['mDice']
        # NaN и None считаем худшими (ставим в конец)
        if dice is None or (isinstance(dice, float) and np.isnan(dice)):
            return -float('inf')
        return dice
    
    all_results.sort(key=sort_key, reverse=True)
    
    # Ограничиваем top_n количеством доступных результатов
    actual_top_n = min(top_n, len(all_results))
    
    # Берем только топ N лучших и худших
    top_correct = all_results[:actual_top_n]
    top_incorrect = all_results[-actual_top_n:] if len(all_results) >= actual_top_n else []
    
    
    # Сохраняем визуализации только для топ N
    vis_dir = os.path.join(output_dir, 'visualizations')
    best_dir = os.path.join(vis_dir, 'best_predictions')
    worst_dir = os.path.join(vis_dir, 'worst_predictions')
    os.makedirs(best_dir, exist_ok=True)
    os.makedirs(worst_dir, exist_ok=True)
    
    
    # Визуализируем только топ N лучших
    for i, result in enumerate(top_correct, 1):
        save_path = os.path.join(best_dir, f"{i:02d}_{result['image_name']}_mDice_{result['metrics']['mDice']:.4f}.png")
        visualize_prediction(
            result['image'], result['pred_mask'], result['gt_mask'],
            result['metrics'], result['image_name'], palette, class_names, save_path
        )
    
    # Визуализируем только топ N худших
    for i, result in enumerate(reversed(top_incorrect), 1):
        save_path = os.path.join(worst_dir, f"{i:02d}_{result['image_name']}_mDice_{result['metrics']['mDice']:.4f}.png")
        visualize_prediction(
            result['image'], result['pred_mask'], result['gt_mask'],
            result['metrics'], result['image_name'], palette, class_names, save_path
        )
    
    logger.info(f"✅ Visualizations saved to {vis_dir}")


def main():
    parser = argparse.ArgumentParser(description='MMSeg test (and eval) a model')
    parser.add_argument('--config', help='train config file path')
    parser.add_argument('--checkpoint', help='checkpoint file')
    parser.add_argument('config_pos', nargs='?', help='train config file path (positional)')
    parser.add_argument('checkpoint_pos', nargs='?', help='checkpoint file (positional)')
    parser.add_argument(
        '--work-dir',
        help=('if specified, the evaluation metric results will be dumped'
              'into the directory as json'))
    parser.add_argument(
        '--data-root', default='datasets/train_dataset_for_students',
        help='Root directory of the dataset')
    parser.add_argument(
        '--split', choices=['train', 'val', 'test'], default='test',
        help='Dataset split to use for testing (default: test)')
    parser.add_argument(
        '--out',
        type=str,
        help='The directory to save output prediction for offline evaluation')
    parser.add_argument(
        '--output-dir',
        type=str,
        help='The directory to save output prediction (alias for --out)')
    parser.add_argument(
        '--save-predictions', action='store_true',
        help='Save prediction results to output directory')
    parser.add_argument(
        '--show', action='store_true', help='show prediction results')
    parser.add_argument(
        '--show-dir',
        help='directory where painted images will be saved. '
        'If specified, it will be automatically saved '
        'to the work_dir/timestamp/show_dir')
    parser.add_argument(
        '--wait-time', type=float, default=2, help='the interval of show (s)')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument(
        '--tta', action='store_true', help='Test time augmentation')
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    parser.add_argument('--exp-name', default='test', help='Experiment name')
    parser.add_argument(
        '--visualize-top-n', type=int, default=0,
        help='Visualize top N best and worst predictions (0 = disabled)')

    args = parser.parse_args()
    
    config_path = args.config or args.config_pos
    checkpoint_path = args.checkpoint or args.checkpoint_pos
    
    if not config_path:
        parser.error('--config (or config as positional argument) is required')
    if not checkpoint_path:
        parser.error('--checkpoint (or checkpoint as positional argument) is required')
    
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    register_all_modules(init_default_scope=True)
    logger.info("="*80)
    logger.info(f"🧪 TESTING: {args.exp_name}")
    logger.info("="*80)
    logger.info(f"⚙️  Config: {config_path}")
    logger.info(f"📦 Checkpoint: {checkpoint_path}")
    logger.info(f"📦 Data Root: {args.data_root}")
    logger.info(f"📊 Split: {args.split}")
    
    # Определяем output_dir для логирования
    output_dir = args.output_dir or args.out
    if output_dir:
        logger.info(f"💾 Output Dir: {output_dir}")
    elif args.work_dir:
        logger.info(f"📁 Work Dir: {args.work_dir}")
    logger.info("="*80)

    cfg = Config.fromfile(config_path)
    cfg.launcher = args.launcher
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # Определяем output_dir для метрик и визуализаций
    # Используем его как work_dir, чтобы временные файлы создавались там, а не в work_dirs
    metrics_output_dir = args.output_dir or args.out
    
    if metrics_output_dir is None:
        # Если output_dir не указан, используем work_dir из аргументов или дефолтный
        if args.work_dir is not None:
            metrics_output_dir = args.work_dir
        elif cfg.get('work_dir', None) is not None:
            metrics_output_dir = cfg.work_dir
        else:
            # Только если ничего не указано, используем дефолтный work_dirs
            metrics_output_dir = os.path.join('./work_dirs',
                                            os.path.splitext(os.path.basename(config_path))[0])
    
    # ВАЖНО: Устанавливаем work_dir ДО создания Runner, чтобы все временные файлы создавались там
    # Если указан --output-dir, то work_dirs не будет создаваться вообще
    cfg.work_dir = metrics_output_dir
    
    logger.info(f"📁 Using work_dir: {cfg.work_dir} (temporary files will be created here)")

    cfg.load_from = checkpoint_path

    cfg = setup_test_config(cfg, args.data_root, args.split)

    if args.show or args.show_dir:
        cfg = trigger_visualization_hook(cfg, args)

    if args.tta:
        cfg.test_dataloader.dataset.pipeline = cfg.tta_pipeline
        cfg.tta_model.module = cfg.model
        cfg.model = cfg.tta_model

    # Определяем output_dir только если явно запрошено сохранение предсказаний
    output_dir = None
    if args.save_predictions:
        output_dir = args.output_dir or args.out
        if output_dir is None:
            output_dir = os.path.join(cfg.work_dir, 'test_results')
        
        os.makedirs(output_dir, exist_ok=True)
        
        # test_evaluator может быть словарем или списком
        if hasattr(cfg, 'test_evaluator'):
            if isinstance(cfg.test_evaluator, dict):
                cfg.test_evaluator['output_dir'] = output_dir
                cfg.test_evaluator['keep_results'] = True
            elif isinstance(cfg.test_evaluator, list) and len(cfg.test_evaluator) > 0:
                # Если список, устанавливаем для первого элемента (обычно это основная метрика)
                cfg.test_evaluator[0]['output_dir'] = output_dir
                cfg.test_evaluator[0]['keep_results'] = True
            logger.info(f"💾 All predictions will be saved to: {output_dir}")
    else:
        # Если не запрошено сохранение всех предсказаний, отключаем сохранение в evaluator
        if hasattr(cfg, 'test_evaluator'):
            if isinstance(cfg.test_evaluator, dict):
                cfg.test_evaluator.pop('output_dir', None)
                cfg.test_evaluator['keep_results'] = False
            elif isinstance(cfg.test_evaluator, list):
                # Удаляем output_dir из всех метрик в списке
                for evaluator in cfg.test_evaluator:
                    if isinstance(evaluator, dict):
                        evaluator.pop('output_dir', None)
                        evaluator['keep_results'] = False
            logger.info("📝 Predictions saving disabled (use --save-predictions to enable)")

    # metrics_output_dir уже установлен выше как cfg.work_dir
    # Просто убеждаемся, что директория существует
    os.makedirs(metrics_output_dir, exist_ok=True)

    runner = Runner.from_cfg(cfg)
    
    # Получаем количество классов для хука
    num_classes = 3
    if hasattr(runner.model, 'decode_head') and hasattr(runner.model.decode_head, 'num_classes'):
        num_classes = runner.model.decode_head.num_classes
    
    # Регистрируем хук для сбора результатов и вычисления метрик покартинно
    # Это позволит избежать второго прохода для визуализации
    per_image_metrics_hook = PerImageMetricsHook(num_classes=num_classes)
    runner.register_hook(per_image_metrics_hook, priority='NORMAL')
    
    # АЖНО: Измеряем время выполнения тестирования
    # Это время включает: загрузку данных, препроцессинг, inference модели,
    # расчет метрик через Evaluator, логирование.
    # Это "End-to-End Throughput", а не чистый "Inference FPS" модели.
    # Для чистого FPS нужно измерять только время внутри model.test_step().
    test_start_time = time.time()
    test_results = runner.test()
    test_end_time = time.time()
    test_duration = test_end_time - test_start_time
    
    
    # Вычисляем глобальные per-class метрики: суммируем площади (intersect/union) по всем изображениям,
    # затем вычисляем метрики из суммарных площадей (правильный подход, эквивалентен evaluator)
    per_class_metrics_from_evaluator = {}
    
    # Вычисляем глобальные per-class метрики из покартинных результатов
    if len(per_image_metrics_hook.results) > 0:
        try:
            
            # Получаем количество классов
            num_classes = per_image_metrics_hook.num_classes
            
            # Инициализируем аккумуляторы для каждого класса
            total_intersect = np.zeros(num_classes, dtype=np.int64)
            total_union = np.zeros(num_classes, dtype=np.int64)
            total_pred = np.zeros(num_classes, dtype=np.int64)
            total_label = np.zeros(num_classes, dtype=np.int64)
            
            # Суммируем площади по всем изображениям для каждого класса
            for result in per_image_metrics_hook.results:
                pred_mask = result['pred_sem_seg']
                gt_mask = result['gt_sem_seg']
                
                # Вычисляем площади для этого изображения
                for class_idx in range(num_classes):
                    pred_class = (pred_mask == class_idx)
                    gt_class = (gt_mask == class_idx)
                    
                    intersect = np.logical_and(pred_class, gt_class).sum()
                    union = np.logical_or(pred_class, gt_class).sum()
                    pred_area = pred_class.sum()
                    label_area = gt_class.sum()
                    
                    total_intersect[class_idx] += intersect
                    total_union[class_idx] += union
                    total_pred[class_idx] += pred_area
                    total_label[class_idx] += label_area
            
            # Вычисляем метрики для каждого класса
            for class_idx in range(num_classes):
                with np.errstate(divide='ignore', invalid='ignore'):
                    iou = total_intersect[class_idx] / total_union[class_idx] if total_union[class_idx] > 0 else 0
                    dice = 2 * total_intersect[class_idx] / (total_pred[class_idx] + total_label[class_idx]) if (total_pred[class_idx] + total_label[class_idx]) > 0 else 0
                    acc = total_intersect[class_idx] / total_label[class_idx] if total_label[class_idx] > 0 else 0
                    
                    per_class_metrics_from_evaluator[class_idx] = {
                        'IoU': float(np.round(np.nan_to_num(iou) * 100, 2)),
                        'Dice': float(np.round(np.nan_to_num(dice) * 100, 2)),
                        'Acc': float(np.round(np.nan_to_num(acc) * 100, 2))
                    }
            
        except Exception as e:
            logger.debug(f"Could not compute per-class metrics from per-image results: {e}")
    
    # Вычисляем FPS и latency (End-to-End метрики)
    inference_metrics = {}
    if hasattr(runner, 'test_dataloader') and runner.test_dataloader is not None:
        try:
            total_samples = len(runner.test_dataloader.dataset)
            if total_samples > 0 and test_duration > 0:
                fps = total_samples / test_duration
                latency_ms = (test_duration / total_samples) * 1000  # средняя latency на изображение
                inference_metrics['fps'] = round(fps, 2)
                inference_metrics['latency_ms'] = round(latency_ms, 2)
                inference_metrics['total_time_s'] = round(test_duration, 2)
                inference_metrics['total_samples'] = total_samples
                inference_metrics['note'] = 'End-to-End throughput (includes data loading, preprocessing, inference, and metric calculation)'
        except Exception:
            pass
    
    if (test_results is None or not isinstance(test_results, dict)) and hasattr(runner, 'message_hub'):
        try:
            test_results = {}
            scalars = runner.message_hub.get_scalar('test')
            if scalars:
                for key, buf in scalars.items():
                    try:
                        test_results[key] = buf.current()
                    except Exception:
                        pass
        except Exception:
            pass

    save_metrics_to_json(
        test_results, 
        metrics_output_dir, 
        config_path, 
        checkpoint_path,
        split=args.split,
        runner=runner,
        cfg=cfg,
        inference_metrics=inference_metrics,
        per_class_metrics_from_evaluator=per_class_metrics_from_evaluator if 'per_class_metrics_from_evaluator' in locals() else {}
    )
    
    # Визуализация топ предсказаний
    if args.visualize_top_n > 0:
        visualize_top_predictions(
            runner=runner,
            top_n=args.visualize_top_n,
            split=args.split,
            output_dir=metrics_output_dir,
            class_names=get_class_names(cfg)
        )

    logger.info("="*80)
    logger.info("✅ Testing completed!")
    logger.info("="*80)
    
    # Явный flush для Jupyter
    sys.stdout.flush()
    sys.stderr.flush()


if __name__ == '__main__':
    try:
        main()
    finally:
        # Гарантируем вывод в Jupyter
        sys.stdout.flush()
        sys.stderr.flush()
