"""Скрипт обучения моделей сегментации с поддержкой ClearML.
"""

import os
import sys
import argparse
from pathlib import Path
import warnings
import torch
import logging

warnings.filterwarnings('ignore', category=UserWarning)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
mmseg_path = os.path.join(project_root, 'mmsegmentation')
if os.path.exists(mmseg_path) and mmseg_path not in sys.path:
    sys.path.insert(0, mmseg_path)
sys.path.append(project_root)

from mmengine.config import Config
from mmengine.hooks import Hook
from mmengine.runner import Runner
from mmseg.utils import register_all_modules

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('TrainScript')

# Загрузка ClearML утилит (только если нужны)
CLEARML_AVAILABLE = False
_clearml_utils_loaded = False

def _load_clearml_utils():
    """Загрузка ClearML утилит."""
    global CLEARML_AVAILABLE, _clearml_utils_loaded
    if _clearml_utils_loaded:
        return CLEARML_AVAILABLE
    
    try:
        from src.utils.clearml_utils import (
            init_clearml_task, log_config_to_clearml, log_metrics_to_clearml
        )
        CLEARML_AVAILABLE = True
        _clearml_utils_loaded = True
        return True
    except ImportError:
        CLEARML_AVAILABLE = False
        _clearml_utils_loaded = True
        return False

# Попытка использовать стандартный EarlyStoppingHook из mmengine
try:
    from mmengine.hooks import EarlyStoppingHook as MMEngineEarlyStoppingHook
    USE_STANDARD_HOOK = True
except ImportError:
    USE_STANDARD_HOOK = False
    MMEngineEarlyStoppingHook = None

class EarlyStoppingHook(Hook):
    """Ранний останов обучения при отсутствии улучшения метрики.
    
    Исправленная версия с правильной логикой поиска метрики и проверки остановки.
    """
    
    def __init__(self, monitor='val/mDice', patience=10, min_delta=0.0, 
                 mode='max', priority='NORMAL'):
        super().__init__()
        self.priority = priority
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best_score = None
        self.counter = 0
        self.best_epoch = 0
        
    def _is_better(self, current, best):
        """Проверяет, лучше ли текущее значение метрики."""
        if best is None:
            return True
        if self.mode == 'max':
            return current > best + self.min_delta
        else:
            return current < best - self.min_delta
    
    def after_val_epoch(self, runner, metrics=None, **kwargs):
        """Обрабатывает результаты валидации и проверяет условие остановки."""
        # Критично: используем только словарь metrics, не лезем в message_hub
        if not metrics:
            return
        
        # Гибкий поиск ключа метрики
        key = self.monitor
        if key not in metrics:
            # Попытка найти без префикса 'val/'
            simple_key = key.split('/')[-1]
            # Попытка найти с префиксом 'val/'
            val_key = f'val/{simple_key}'
            
            if simple_key in metrics:
                key = simple_key
            elif val_key in metrics:
                key = val_key
            else:
                # Метрика не найдена - выходим без ошибки
                # (возможно, метрика еще не вычислена или называется по-другому)
                return
        
        score = float(metrics[key])
        
        # Логика сравнения и обновления счетчика
        if self._is_better(score, self.best_score):
            self.best_score = score
            self.best_epoch = runner.epoch
            self.counter = 0  # Сброс счетчика при улучшении
        else:
            self.counter += 1  # Увеличение счетчика при отсутствии улучшения
        
        # Проверка условия остановки (вынесена из else для надежности)
        if self.counter >= self.patience:
            # Логируем только на главном процессе (для распределенного обучения)
            if not hasattr(runner, 'rank') or runner.rank == 0:
                logger.warning('\n🛑 EarlyStopping: Stop training!')
                logger.warning(f'   Metric {self.monitor} did not improve for {self.patience} epochs.')
                logger.warning(f'   Best score: {self.best_score:.4f} at epoch {self.best_epoch}')
            runner.should_stop = True

class ClearMLHook(Hook):
    """Логирование train/val метрик в ClearML с защитой от зависаний."""

    def __init__(self, clearml_task, priority='NORMAL'):
        super().__init__()
        self.priority = priority
        self.task = clearml_task
        self.enabled = CLEARML_AVAILABLE and (self.task is not None)
        self.failed_attempts = 0
        self.max_failed_attempts = 5  # После 5 неудачных попыток отключаем ClearML
        
        # Отключаем автоматическое обновление задачи, которое может зависнуть
        if self.task is not None:
            try:
                # Устанавливаем режим, при котором задача не обновляется автоматически
                # Обновления будут только через наш хук
                import os
                if os.environ.get('CLEARML_OFFLINE_MODE', '').lower() not in ('1', 'true', 'yes'):
                    # В онлайн режиме пытаемся отключить автообновление
                    # Но если это не работает - просто продолжаем с защитой от зависания
                    pass
            except Exception:
                pass

    def after_train_iter(self, runner, batch_idx, data_batch=None, outputs=None, **kwargs):
        if not self.enabled or self.failed_attempts >= self.max_failed_attempts:
            return
        
        try:
            metrics = {}
            scalars = getattr(runner.message_hub, 'log_scalars', None)
            if scalars:
                for key, buf in scalars.items():
                    if not key.startswith('train/'):
                        continue
                    try:
                        metrics[key] = float(buf.current())
                    except Exception:
                        pass
            if metrics:
                from src.utils.clearml_utils import log_metrics_to_clearml
                log_metrics_to_clearml(self.task, metrics, step=runner.iter)
        except Exception as e:
            self.failed_attempts += 1
            if self.failed_attempts >= self.max_failed_attempts:
                logger.warning(f"ClearML logging disabled after {self.max_failed_attempts} failures")

    def after_val_epoch(self, runner, metrics=None, **kwargs):
        if not self.enabled or not metrics or self.failed_attempts >= self.max_failed_attempts:
            return
        
        try:
            log_metrics = {f'val/{k}': float(v) for k, v in metrics.items()}
            from src.utils.clearml_utils import log_metrics_to_clearml
            log_metrics_to_clearml(self.task, log_metrics, epoch=runner.epoch)
        except Exception as e:
            self.failed_attempts += 1
            if self.failed_attempts >= self.max_failed_attempts:
                logger.warning(f"ClearML logging disabled after {self.max_failed_attempts} failures")

def setup_clearml(config_path, args):
    if args.no_clearml:
        return None
    
    # Ленивая загрузка ClearML утилит
    if not _load_clearml_utils() or not CLEARML_AVAILABLE:
        return None
    
    try:
        from src.utils.clearml_utils import init_clearml_task, log_config_to_clearml
        
        config_name = Path(config_path).stem if config_path else 'experiment'
        
        task = init_clearml_task(
            project_name=args.clearml_project or 'MMSegmentation',
            task_name=args.clearml_task_name or f'{args.exp_name}_{config_name}',
            tags=(args.clearml_tags.split(',') if args.clearml_tags else [])
        )
        
        if not task:
            return None
        
        if config_path and task:
            log_config_to_clearml(task, config_path)
        
        return task
    except Exception as e:
        logger.warning(f"ClearML setup failed: {e}")
        return None

def validate_setup(data_root, work_dir, device):
    if not Path(data_root).exists():
        raise ValueError(f"No dataset: {data_root}")
    Path(work_dir).mkdir(parents=True, exist_ok=True)

def main():
    try:
        from IPython import get_ipython
        is_jupyter = get_ipython() is not None
    except ImportError:
        is_jupyter = False
    
    if not is_jupyter:
        import mmengine.logging
        
        _orig_print_log = mmengine.logging.print_log
        
        def patched_print_log(msg, logger=None, level=logging.INFO):
            msg_str = str(msg)
            if "unexpected key in source state_dict" in msg_str and "fc.weight" in msg_str: return
            if "FileClient" in msg_str and "deprecated" in msg_str: return
            if "HardDiskBackend" in msg_str: return
            _orig_print_log(msg, logger, level)
            
        mmengine.logging.print_log = patched_print_log

        _orig_emit = logging.StreamHandler.emit

        def patched_emit(self, record):
            try:
                msg = self.format(record)
                if "unexpected key" in msg and "fc.weight" in msg: return
                if "FileClient" in msg and "deprecated" in msg: return
                if "HardDiskBackend" in msg: return
            except: 
                pass
            _orig_emit(self, record)

        logging.StreamHandler.emit = patched_emit

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--work-dir', required=True)
    parser.add_argument('--data-root', default='datasets/train_dataset_for_students')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--exp-name', default='default')
    parser.add_argument('--clearml-project', default=None)
    parser.add_argument('--clearml-task-name', default=None)
    parser.add_argument('--clearml-tags', default=None)
    parser.add_argument('--clearml-output-uri', default=None)
    parser.add_argument('--no-clearml', action='store_true')
    parser.add_argument('--early-stopping-patience', type=int, default=None)
    parser.add_argument('--early-stopping-metric', type=str, default='val/mDice')
    parser.add_argument('--early-stopping-min-delta', type=float, default=0.0)
    parser.add_argument('--early-stopping-mode', type=str, default='max', choices=['max', 'min'])

    args = parser.parse_args()
    validate_setup(args.data_root, args.work_dir, args.device)

    # ClearML с защитой от зависания
    clearml_task = None
    if not args.no_clearml:
        try:
            clearml_task = setup_clearml(args.config, args)
        except Exception as e:
            logger.warning(f"ClearML initialization failed: {e}")
            clearml_task = None
        else:
            if clearml_task is None:
                logger.warning("ClearML не инициализирован (продолжаем без логирования)")
    
    register_all_modules(init_default_scope=True)
    cfg = Config.fromfile(args.config)
    cfg.work_dir = args.work_dir
    cfg.randomness = dict(seed=args.seed, deterministic=False)
    
    if hasattr(cfg, 'train_dataloader'):
        cfg.train_dataloader.batch_size = args.batch_size
        cfg.train_dataloader.dataset.data_root = args.data_root
        cfg.val_dataloader.dataset.data_root = args.data_root
        cfg.test_dataloader.dataset.data_root = args.data_root
    
    if hasattr(cfg, 'train_cfg'):
        cfg.train_cfg.max_epochs = args.epochs
    
    if hasattr(cfg, 'optim_wrapper') and hasattr(cfg.optim_wrapper, 'optimizer'):
        cfg.optim_wrapper.optimizer['lr'] = args.lr
        cfg.optim_wrapper.optimizer['weight_decay'] = args.weight_decay

    use_pretty_printer = False
    try:
        from src.utils.pretty_printer import PrettyPrintHook
        use_pretty_printer = True
    except ImportError:
        pass

    if use_pretty_printer and hasattr(cfg, 'default_hooks'):
        if 'logger' in cfg.default_hooks:
            cfg.default_hooks['logger']['interval'] = 1000000
        else:
            cfg.default_hooks['logger'] = dict(type='LoggerHook', interval=1000000)

    runner = Runner.from_cfg(cfg)

    if use_pretty_printer:
        runner.register_hook(PrettyPrintHook(priority='NORMAL', update_interval=10))
    
    if clearml_task:
        runner.register_hook(ClearMLHook(clearml_task))
    
    if args.early_stopping_patience is not None:
        # Используем стандартный хук из mmengine, если доступен
        if USE_STANDARD_HOOK:
            # Стандартный хук использует 'rule' вместо 'mode'
            # Преобразуем 'max' -> 'greater', 'min' -> 'less'
            rule_map = {'max': 'greater', 'min': 'less'}
            rule = rule_map.get(args.early_stopping_mode, 'greater')
            early_stopping_hook = MMEngineEarlyStoppingHook(
                monitor=args.early_stopping_metric,
                patience=args.early_stopping_patience,
                min_delta=args.early_stopping_min_delta,
                rule=rule
            )
            logger.info("Using standard EarlyStoppingHook from mmengine")
        else:
            # Используем исправленный кастомный хук
            early_stopping_hook = EarlyStoppingHook(
                monitor=args.early_stopping_metric,
                patience=args.early_stopping_patience,
                min_delta=args.early_stopping_min_delta,
                mode=args.early_stopping_mode
            )
            logger.info("Using custom EarlyStoppingHook")
        runner.register_hook(early_stopping_hook)

    logger.info("="*80)
    logger.info(f"🚀 EXPERIMENT: {args.exp_name}")
    logger.info("="*80)
    
    try:
        if hasattr(cfg.model, 'decode_head'):
            losses = cfg.model.decode_head.get('loss_decode', [])
            for loss in losses:
                if loss.get('type') == 'CrossEntropyLoss':
                    class_weight = loss.get('class_weight', None)
                    if class_weight:
                        logger.info(f"📊 Class Weights: {class_weight}")
                        for idx, weight in enumerate(class_weight):
                            logger.info(f"   Class {idx}: {weight:.4f}")
                        break
    except Exception:
        pass
    
    logger.info("-"*80)
    logger.info(f"Config: {args.config}")
    logger.info(f"Work Dir: {args.work_dir}")
    logger.info(f"Data Root: {args.data_root}")
    logger.info(f"Epochs: {args.epochs} | Batch Size: {args.batch_size}")
    logger.info(f"LR: {args.lr} | Weight Decay: {args.weight_decay}")
    logger.info(f"Seed: {args.seed} | Device: {args.device}")
    if args.early_stopping_patience is not None:
        logger.info(f"Early Stopping: {args.early_stopping_metric}, patience={args.early_stopping_patience}")
    logger.info("="*80)
    logger.info("Starting training...")
    logger.info("="*80)
    
    # Обучение с защитой от падения
    try:
        runner.train()
        logger.info("="*80)
        logger.info("✅ Training completed successfully!")
        logger.info("="*80)
    except KeyboardInterrupt:
        logger.warning("⚠️  Training interrupted by user (Ctrl+C)")
    except Exception as e:
        logger.error(f"❌ Training failed with error: {e}")
        raise
    finally:
        # Гарантируем закрытие ClearML задачи даже при падении
        if clearml_task is not None:
            try:
                clearml_task.close()
            except Exception:
                pass

if __name__ == '__main__':
    main()
