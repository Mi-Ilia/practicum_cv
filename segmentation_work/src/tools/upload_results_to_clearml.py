"""Скрипт для загрузки результатов экспериментов в ClearML.

Используется для загрузки результатов после завершения обучения,
если ClearML не успел залогировать их во время обучения.

Загружает:
- Чекпоинты (.pth)
- Метрики (scalars.json, test_metrics.json)
- Логи (.log)
- Визуализации (best/worst predictions)
- Конфигурацию

Примеры использования:

1. Загрузить результаты в новую задачу:
   python src/tools/upload_results_to_clearml.py \
       --exp-dir experiments/h1_experiment \
       --project practicum_segmentation \
       --task-name h1_deeplabv3plus_results \
       --tags "H1,DeepLabV3+"

2. Загрузить только метрики валидации:
   python src/tools/upload_results_to_clearml.py \
       --exp-dir experiments/h1_experiment \
       --project practicum_segmentation \
       --task-name h1_deeplabv3plus_results \
       --skip-checkpoints --skip-visualizations

3. Загрузить результаты в существующую задачу:
   python src/tools/upload_results_to_clearml.py \
       --exp-dir experiments/h1_experiment \
       --task-id <clearml_task_id>
"""

import os
import sys
import argparse
import json
from pathlib import Path
from typing import Optional, Dict, Any
import logging
import time
import signal
import threading
from contextlib import contextmanager

# Добавляем project root в path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(project_root)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('UploadToClearML')


def format_size(size_bytes: int) -> str:
    """Форматирует размер файла в читаемый вид."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} TB"


@contextmanager
def timeout_handler(timeout_seconds: int):
    """Контекстный менеджер для обработки таймаутов."""
    def timeout_signal(signum, frame):
        raise TimeoutError(f"Operation timed out after {timeout_seconds} seconds")
    
    if hasattr(signal, 'SIGALRM'):  # Unix only
        old_handler = signal.signal(signal.SIGALRM, timeout_signal)
        signal.alarm(timeout_seconds)
        try:
            yield
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
    else:
        # Windows doesn't support SIGALRM, just yield without timeout
        yield

try:
    from clearml import Task
    CLEARML_AVAILABLE = True
except ImportError:
    CLEARML_AVAILABLE = False
    logger.error("ClearML не установлен. Установите: pip install clearml")
    sys.exit(1)


def find_files_by_pattern(directory: Path, pattern: str) -> list:
    """Находит файлы по паттерну в директории."""
    return list(directory.glob(pattern))


def upload_checkpoints(task: Task, exp_dir: Path):
    """Загружает чекпоинты модели в ClearML."""
    logger.info("Uploading checkpoints...")
    sys.stdout.flush()
    
    # Ищем best чекпоинты
    best_checkpoints = list(exp_dir.glob('best_*.pth'))
    
    if not best_checkpoints:
        logger.info("   No checkpoints found")
        return
    
    total_size = sum(f.stat().st_size for f in best_checkpoints)
    logger.info(f"   Found {len(best_checkpoints)} checkpoint(s), total size: {format_size(total_size)}")
    sys.stdout.flush()
    
    uploaded = 0
    failed = 0
    
    def upload_single_checkpoint(ckpt_path, ckpt_name):
        """Загружает один чекпоинт с таймаутом."""
        try:
            task.upload_artifact(
                name=f'checkpoint_{ckpt_path.stem}',
                artifact_object=str(ckpt_path)
            )
            return True
        except Exception as e:
            logger.error(f"   Upload error: {e}")
            return False
    
    for idx, ckpt in enumerate(best_checkpoints, 1):
        ckpt_size = ckpt.stat().st_size
        logger.info(f"   [{idx}/{len(best_checkpoints)}] Uploading: {ckpt.name} ({format_size(ckpt_size)})...")
        sys.stdout.flush()
        start_time = time.time()
        
        upload_success = False
        upload_error = None
        
        def upload_thread():
            nonlocal upload_success, upload_error
            try:
                upload_success = upload_single_checkpoint(ckpt, ckpt.name)
            except Exception as e:
                upload_error = e
                upload_success = False
        
        upload_th = threading.Thread(target=upload_thread, daemon=True)
        upload_th.start()
        upload_th.join(timeout=300)  # Таймаут 5 минут для больших файлов
        
        if upload_th.is_alive():
            logger.error(f"   [FAIL] [{idx}/{len(best_checkpoints)}] Timeout uploading {ckpt.name} (exceeded 5 minutes)")
            sys.stdout.flush()
            failed += 1
        elif upload_success:
            elapsed = time.time() - start_time
            logger.info(f"   [OK] [{idx}/{len(best_checkpoints)}] Completed: {ckpt.name} (took {elapsed:.1f}s)")
            sys.stdout.flush()
            uploaded += 1
        else:
            error_msg = str(upload_error) if upload_error else "Unknown error"
            logger.error(f"   [FAIL] [{idx}/{len(best_checkpoints)}] Failed: {ckpt.name} - {error_msg}")
            sys.stdout.flush()
            failed += 1
    
    # Загружаем last_checkpoint как параметр
    last_ckpt_file = exp_dir / 'last_checkpoint'
    if last_ckpt_file.exists():
        try:
            with open(last_ckpt_file, 'r') as f:
                last_ckpt_name = f.read().strip()
            task.set_parameter('last_checkpoint', last_ckpt_name)
            logger.info(f"   Set last_checkpoint parameter: {last_ckpt_name}")
        except Exception as e:
            logger.warning(f"   Failed to set last_checkpoint parameter: {e}")
    
    logger.info(f"   Summary: {uploaded} uploaded, {failed} failed")


def upload_metrics(task: Task, exp_dir: Path, skip_training_metrics: bool = False, batch_size: int = 50):
    """Загружает метрики в ClearML.
    
    Args:
        task: ClearML Task объект
        exp_dir: Директория эксперимента
        skip_training_metrics: Если True, пропускает загрузку scalars.json (метрики обучения)
        batch_size: Размер батча метрик перед flush (ускоряет отправку при нестабильной сети)
    """
    logger.info("Uploading metrics...")
    sys.stdout.flush()
    
    uploaded_count = 0
    failed_count = 0
    
    # 1. Загружаем scalars.json (метрики обучения)
    if not skip_training_metrics:
        scalars_files = list(exp_dir.glob('*/vis_data/scalars.json'))
        if scalars_files:
            logger.info(f"   Found {len(scalars_files)} scalars.json file(s)")
            sys.stdout.flush()
        
        for scalars_file in scalars_files:
            log_dir = scalars_file.parent
            try:
                logger.info(f"   Processing: {log_dir.parent.name}/scalars.json...")
                sys.stdout.flush()
                start_time = time.time()
                
                with open(scalars_file, 'r') as f:
                    lines = f.readlines()
                
                metrics_count = 0
                total_lines = len(lines)
                logger.info(f"   Reading {total_lines} metric entries...")
                sys.stdout.flush()
                
                # Scalars.json в формате JSON Lines
                # Важно: в некоторых логгерах train-метрики идут с step=iter,
                # а val-метрики — с step=epoch. Чтобы графики в ClearML не "съезжали",
                # мы маппим val-метрики на последнюю train-итерацию перед валидацией.
                last_train_iteration: Optional[int] = None
                for line_idx, line in enumerate(lines, 1):
                    try:
                        data = json.loads(line.strip())
                        # Логируем метрики с правильной эпохой/итерацией
                        if 'step' in data:
                            step = data['step']
                            # Обновляем "текущую" train-итерацию (если она есть в записи).
                            # В mmengine/mmseg обычно присутствует 'iter' для train-строк.
                            if 'iter' in data:
                                try:
                                    last_train_iteration = int(data['iter'])
                                except Exception:
                                    # На крайний случай пробуем step
                                    try:
                                        last_train_iteration = int(step)
                                    except Exception:
                                        last_train_iteration = last_train_iteration

                            is_val_record = any(isinstance(k, str) and k.startswith('val/') for k in data.keys())
                            # Для val/* используем последнюю train-итерацию, если она известна.
                            try:
                                iteration = int(last_train_iteration) if (is_val_record and last_train_iteration is not None) else int(step)
                            except Exception:
                                iteration = step

                            for key, value in data.items():
                                if key in ['step', 'time']:
                                    continue
                                
                                # Определяем категорию
                                if 'loss' in key.lower():
                                    title = 'Losses'
                                elif 'iou' in key.lower() or 'dice' in key.lower():
                                    title = 'Metrics'
                                else:
                                    title = 'Other'
                                
                                series = key
                                
                                # Пытаемся загрузить метрику с обработкой ошибок
                                try:
                                    task.get_logger().report_scalar(
                                        title=title,
                                        series=series,
                                        value=float(value),
                                        iteration=iteration
                                    )
                                    metrics_count += 1
                                except Exception as metric_error:
                                    # Если ошибка сети - логируем и продолжаем
                                    if 'SSL' in str(metric_error) or 'timeout' in str(metric_error).lower() or 'connection' in str(metric_error).lower():
                                        logger.warning(f"   Network error at step {step}, metric {key}: {metric_error}. Continuing...")
                                        sys.stdout.flush()
                                        # Небольшая пауза перед следующей попыткой
                                        time.sleep(0.1)
                                        # Пытаемся еще раз
                                        try:
                                            task.get_logger().report_scalar(
                                                title=title,
                                                series=series,
                                                value=float(value),
                                                iteration=iteration
                                            )
                                            metrics_count += 1
                                        except Exception:
                                            # Если и вторая попытка не удалась - пропускаем эту метрику
                                            pass
                                    else:
                                        # Другие ошибки - просто пропускаем
                                        pass
                        
                        # Показываем прогресс каждые 10 записей (для метрик обучения)
                        if line_idx % 10 == 0:
                            logger.info(f"   Progress: {line_idx}/{total_lines} entries, {metrics_count} metrics uploaded...")
                            sys.stdout.flush()
                        # Принудительный flush каждые 50 метрик
                        if metrics_count > 0 and metrics_count % max(1, int(batch_size)) == 0:
                            try:
                                task.get_logger().flush()
                            except Exception:
                                pass
                            sys.stdout.flush()
                    except Exception:
                        continue
                
                # Финальный flush для этого лога (важно для val/* точек в конце файла)
                try:
                    task.get_logger().flush()
                except Exception:
                    pass

                elapsed = time.time() - start_time
                logger.info(f"   [OK] Uploaded {metrics_count} metric points from {log_dir.parent.name} (took {elapsed:.1f}s)")
                sys.stdout.flush()
                uploaded_count += 1
            except Exception as e:
                logger.error(f"   [FAIL] Failed to upload {scalars_file}: {e}")
                sys.stdout.flush()
                failed_count += 1
    else:
        logger.info("   Skipping training metrics (scalars.json) as requested")
        sys.stdout.flush()
    
    # 2. Загружаем test_metrics.json (метрики тестирования)
    results_dirs = list(exp_dir.glob('*_results'))
    if results_dirs:
        logger.info(f"   Found {len(results_dirs)} results directory(ies)")
        sys.stdout.flush()
    
    for results_dir in results_dirs:
        metrics_file = results_dir / 'test_metrics.json'
        if metrics_file.exists():
            try:
                logger.info(f"   Processing: {results_dir.name}/test_metrics.json...")
                sys.stdout.flush()
                start_time = time.time()
                
                with open(metrics_file, 'r') as f:
                    data = json.load(f)
                
                # Загружаем как артефакт
                task.upload_artifact(
                    name=f'test_metrics_{results_dir.name}',
                    artifact_object=data
                )
                
                # Также логируем метрики как скаляры
                metrics_logged = 0
                if 'metrics' in data and 'main' in data['metrics']:
                    for metric_name, metric_value in data['metrics']['main'].items():
                        task.get_logger().report_single_value(
                            name=f'Test/{metric_name}',
                            value=float(metric_value)
                        )
                        metrics_logged += 1
                
                try:
                    task.get_logger().flush()
                except Exception:
                    pass
                
                elapsed = time.time() - start_time
                logger.info(f"   [OK] Uploaded test metrics from {results_dir.name} ({metrics_logged} metrics, took {elapsed:.1f}s)")
                sys.stdout.flush()
                uploaded_count += 1
            except Exception as e:
                logger.error(f"   [FAIL] Failed to upload {metrics_file}: {e}")
                sys.stdout.flush()
                failed_count += 1
    
    if uploaded_count == 0:
        logger.warning("   No metrics files found")
    else:
        logger.info(f"   Summary: {uploaded_count} uploaded, {failed_count} failed")
    sys.stdout.flush()


def upload_visualizations(task: Task, exp_dir: Path):
    """Загружает визуализации в ClearML."""
    logger.info("Uploading visualizations...")
    sys.stdout.flush()
    
    uploaded_count = 0
    failed_count = 0
    
    # Ищем визуализации в val_results и test_results
    results_dirs = list(exp_dir.glob('*_results'))
    
    total_images = 0
    for results_dir in results_dirs:
        vis_dir = results_dir / 'visualizations'
        if vis_dir.exists():
            best_dir = vis_dir / 'best_predictions'
            worst_dir = vis_dir / 'worst_predictions'
            if best_dir.exists():
                total_images += len(list(best_dir.glob('*.png'))[:5])
            if worst_dir.exists():
                total_images += len(list(worst_dir.glob('*.png'))[:5])
    
    if total_images == 0:
        logger.info("   No visualizations found")
        return
    
    logger.info(f"   Found {total_images} image(s) to upload")
    sys.stdout.flush()
    
    current_idx = 0
    for results_dir in results_dirs:
        vis_dir = results_dir / 'visualizations'
        if not vis_dir.exists():
            continue
        
        # Загружаем best predictions
        best_dir = vis_dir / 'best_predictions'
        if best_dir.exists():
            best_images = sorted(best_dir.glob('*.png'))[:5]
            for img_path in best_images:
                current_idx += 1
                img_size = img_path.stat().st_size
                logger.info(f"   [{current_idx}/{total_images}] Uploading: {results_dir.name}/best/{img_path.name} ({format_size(img_size)})...")
                sys.stdout.flush()
                start_time = time.time()
                
                upload_success = False
                upload_error = None
                
                def upload_image():
                    nonlocal upload_success, upload_error
                    try:
                        task.get_logger().report_image(
                            title=f'{results_dir.name}/Best Predictions',
                            series=img_path.stem,
                            iteration=0,
                            local_path=str(img_path)
                        )
                        upload_success = True
                    except Exception as e:
                        upload_error = e
                        upload_success = False
                
                upload_th = threading.Thread(target=upload_image, daemon=True)
                upload_th.start()
                upload_th.join(timeout=60)  # Таймаут 1 минута для изображений
                
                if upload_th.is_alive():
                    logger.error(f"   [FAIL] [{current_idx}/{total_images}] Timeout uploading {img_path.name} (exceeded 1 minute)")
                    sys.stdout.flush()
                    failed_count += 1
                elif upload_success:
                    elapsed = time.time() - start_time
                    logger.info(f"   [OK] [{current_idx}/{total_images}] Completed: {img_path.name} (took {elapsed:.1f}s)")
                    sys.stdout.flush()
                    uploaded_count += 1
                else:
                    error_msg = str(upload_error) if upload_error else "Unknown error"
                    logger.error(f"   [FAIL] [{current_idx}/{total_images}] Failed: {img_path.name} - {error_msg}")
                    sys.stdout.flush()
                    failed_count += 1
        
        # Загружаем worst predictions
        worst_dir = vis_dir / 'worst_predictions'
        if worst_dir.exists():
            worst_images = sorted(worst_dir.glob('*.png'))[:5]
            for img_path in worst_images:
                current_idx += 1
                img_size = img_path.stat().st_size
                logger.info(f"   [{current_idx}/{total_images}] Uploading: {results_dir.name}/worst/{img_path.name} ({format_size(img_size)})...")
                sys.stdout.flush()
                start_time = time.time()
                
                upload_success = False
                upload_error = None
                
                def upload_image():
                    nonlocal upload_success, upload_error
                    try:
                        task.get_logger().report_image(
                            title=f'{results_dir.name}/Worst Predictions',
                            series=img_path.stem,
                            iteration=0,
                            local_path=str(img_path)
                        )
                        upload_success = True
                    except Exception as e:
                        upload_error = e
                        upload_success = False
                
                upload_th = threading.Thread(target=upload_image, daemon=True)
                upload_th.start()
                upload_th.join(timeout=60)  # Таймаут 1 минута для изображений
                
                if upload_th.is_alive():
                    logger.error(f"   [FAIL] [{current_idx}/{total_images}] Timeout uploading {img_path.name} (exceeded 1 minute)")
                    sys.stdout.flush()
                    failed_count += 1
                elif upload_success:
                    elapsed = time.time() - start_time
                    logger.info(f"   [OK] [{current_idx}/{total_images}] Completed: {img_path.name} (took {elapsed:.1f}s)")
                    sys.stdout.flush()
                    uploaded_count += 1
                else:
                    error_msg = str(upload_error) if upload_error else "Unknown error"
                    logger.error(f"   [FAIL] [{current_idx}/{total_images}] Failed: {img_path.name} - {error_msg}")
                    sys.stdout.flush()
                    failed_count += 1
    
    logger.info(f"   Summary: {uploaded_count} uploaded, {failed_count} failed")
    sys.stdout.flush()


def upload_config(task: Task, exp_dir: Path):
    """Загружает конфигурацию в ClearML."""
    logger.info("Uploading configuration...")
    sys.stdout.flush()
    
    # Ищем конфиг в директории эксперимента
    config_files = list(exp_dir.glob('*.py'))
    
    if not config_files:
        logger.info("   No config files found")
        return
    
    logger.info(f"   Found {len(config_files)} config file(s)")
    sys.stdout.flush()
    
    uploaded = 0
    failed = 0
    
    def upload_single_config(config_path, config_name):
        """Загружает один конфиг с таймаутом."""
        try:
            task.upload_artifact(
                name=f'config_{config_path.stem}',
                artifact_object=str(config_path)
            )
            return True
        except Exception as e:
            logger.error(f"   Upload error: {e}")
            return False
    
    for idx, config_file in enumerate(config_files, 1):
        config_size = config_file.stat().st_size
        logger.info(f"   [{idx}/{len(config_files)}] Uploading: {config_file.name} ({format_size(config_size)})...")
        sys.stdout.flush()
        start_time = time.time()
        
        upload_success = False
        upload_error = None
        
        def upload_thread():
            nonlocal upload_success, upload_error
            try:
                upload_success = upload_single_config(config_file, config_file.name)
            except Exception as e:
                upload_error = e
                upload_success = False
        
        upload_th = threading.Thread(target=upload_thread, daemon=True)
        upload_th.start()
        upload_th.join(timeout=60)  # Таймаут 1 минута для конфигов
        
        if upload_th.is_alive():
            logger.error(f"   [FAIL] [{idx}/{len(config_files)}] Timeout uploading {config_file.name} (exceeded 1 minute)")
            sys.stdout.flush()
            failed += 1
        elif upload_success:
            elapsed = time.time() - start_time
            logger.info(f"   [OK] [{idx}/{len(config_files)}] Completed: {config_file.name} (took {elapsed:.1f}s)")
            sys.stdout.flush()
            uploaded += 1
        else:
            error_msg = str(upload_error) if upload_error else "Unknown error"
            logger.error(f"   [FAIL] [{idx}/{len(config_files)}] Failed: {config_file.name} - {error_msg}")
            sys.stdout.flush()
            failed += 1
    
    logger.info(f"   Summary: {uploaded} uploaded, {failed} failed")
    sys.stdout.flush()


def upload_logs(task: Task, exp_dir: Path):
    """Загружает логи в ClearML."""
    logger.info("Uploading logs...")
    sys.stdout.flush()
    
    # Собираем все логи
    all_log_files = []
    for log_dir in exp_dir.glob('*/'):
        log_files = list(log_dir.glob('*.log'))
        all_log_files.extend([(log_file, log_dir.name) for log_file in log_files])
    
    if not all_log_files:
        logger.info("   No log files found")
        return
    
    total_size = sum(f[0].stat().st_size for f in all_log_files)
    logger.info(f"   Found {len(all_log_files)} log file(s), total size: {format_size(total_size)}")
    sys.stdout.flush()
    
    uploaded = 0
    failed = 0
    
    def upload_single_log(log_path, log_name):
        """Загружает один лог с таймаутом."""
        try:
            task.upload_artifact(
                name=f'log_{log_name}',
                artifact_object=str(log_path)
            )
            return True
        except Exception as e:
            logger.error(f"   Upload error: {e}")
            return False
    
    for idx, (log_file, log_dir_name) in enumerate(all_log_files, 1):
        log_size = log_file.stat().st_size
        logger.info(f"   [{idx}/{len(all_log_files)}] Uploading: {log_dir_name}/{log_file.name} ({format_size(log_size)})...")
        sys.stdout.flush()
        start_time = time.time()
        
        upload_success = False
        upload_error = None
        
        def upload_thread():
            nonlocal upload_success, upload_error
            try:
                upload_success = upload_single_log(log_file, log_dir_name)
            except Exception as e:
                upload_error = e
                upload_success = False
        
        upload_th = threading.Thread(target=upload_thread, daemon=True)
        upload_th.start()
        upload_th.join(timeout=120)  # Таймаут 2 минуты для логов
        
        if upload_th.is_alive():
            logger.error(f"   [FAIL] [{idx}/{len(all_log_files)}] Timeout uploading {log_file.name} (exceeded 2 minutes)")
            sys.stdout.flush()
            failed += 1
        elif upload_success:
            elapsed = time.time() - start_time
            logger.info(f"   [OK] [{idx}/{len(all_log_files)}] Completed: {log_file.name} (took {elapsed:.1f}s)")
            sys.stdout.flush()
            uploaded += 1
        else:
            error_msg = str(upload_error) if upload_error else "Unknown error"
            logger.error(f"   [FAIL] [{idx}/{len(all_log_files)}] Failed: {log_file.name} - {error_msg}")
            sys.stdout.flush()
            failed += 1
    
    logger.info(f"   Summary: {uploaded} uploaded, {failed} failed")
    sys.stdout.flush()


def main():
    parser = argparse.ArgumentParser(
        description='Upload experiment results to ClearML',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        '--exp-dir',
        required=True,
        help='Path to experiment directory (e.g., experiments/h1_experiment)'
    )
    parser.add_argument(
        '--project',
        help='ClearML project name (required if creating new task)'
    )
    parser.add_argument(
        '--task-name',
        help='ClearML task name (required if creating new task)'
    )
    parser.add_argument(
        '--task-id',
        help='Existing ClearML task ID to update (alternative to --project/--task-name)'
    )
    parser.add_argument(
        '--tags',
        help='Comma-separated tags (e.g., "H1,DeepLabV3+")'
    )
    parser.add_argument(
        '--skip-checkpoints',
        action='store_true',
        help='Skip uploading checkpoints'
    )
    parser.add_argument(
        '--skip-visualizations',
        action='store_true',
        help='Skip uploading visualizations'
    )
    parser.add_argument(
        '--skip-logs',
        action='store_true',
        help='Skip uploading logs'
    )
    parser.add_argument(
        '--skip-config',
        action='store_true',
        help='Skip uploading config'
    )
    parser.add_argument(
        '--skip-training-metrics',
        action='store_true',
        help='Skip uploading training metrics (scalars.json). Use this if network is unstable to avoid hanging.'
    )
    parser.add_argument(
        '--offline-mode',
        action='store_true',
        help='Use ClearML offline mode. Metrics will be saved locally and synced later when network is stable.'
    )
    parser.add_argument(
        '--deferred-init',
        action=argparse.BooleanOptionalAction,
        default=False,
        help='Use ClearML deferred Task.init (can help on unstable network). Default: disabled.'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=50,
        help='Number of metrics to batch before flushing to ClearML (default: 50). Lower values = more frequent uploads but slower.'
    )
    
    args = parser.parse_args()
    
    # Проверка параметров
    if not args.task_id and (not args.project or not args.task_name):
        parser.error('Either --task-id or both --project and --task-name are required')
    
    exp_dir = Path(args.exp_dir)
    if not exp_dir.exists():
        logger.error(f"Experiment directory not found: {exp_dir}")
        sys.exit(1)
    
    logger.info("="*80)
    logger.info("Uploading experiment results to ClearML")
    logger.info("="*80)
    logger.info(f"Experiment directory: {exp_dir}")
    
    # Устанавливаем офлайн режим если запрошен
    if args.offline_mode:
        os.environ['CLEARML_OFFLINE_MODE'] = '1'
        logger.info("Offline mode enabled: metrics will be saved locally and synced later")
        sys.stdout.flush()
    
    # Настраиваем ClearML для предотвращения зависания
    os.environ['CLEARML_MONITOR_ITERATION_REPORTING'] = '0'
    os.environ['CLEARML_MONITOR_GPU'] = '0'
    
    # Устанавливаем таймауты для HTTP запросов ClearML
    os.environ['CLEARML_API_DEFAULT_TIMEOUT'] = '30'
    os.environ['CLEARML_API_MAX_RETRIES_ON_ERROR'] = '3'
    
    # Инициализация или подключение к задаче
    task = None
    try:
        if args.task_id:
            logger.info(f"Connecting to existing task: {args.task_id}...")
            sys.stdout.flush()
            task = Task.get_task(task_id=args.task_id)
            logger.info(f"Connected to task: {task.name} (ID: {task.id})")
            sys.stdout.flush()
        else:
            logger.info(f"Creating new task: {args.project}/{args.task_name}...")
            logger.info("   Note: This may take a moment if network is slow...")
            sys.stdout.flush()
            tags = args.tags.split(',') if args.tags else None

            task = Task.init(
                project_name=args.project,
                task_name=args.task_name,
                tags=tags,
                reuse_last_task_id=False,
                auto_connect_streams=False,
                auto_connect_arg_parser=False,
                deferred_init=bool(args.deferred_init),
            )
            
            logger.info(f"Task created successfully!")
            logger.info(f"   Task name: {task.name}")
            logger.info(f"   Task ID: {task.id}")
            logger.info(f"   Project: {args.project}")
            if tags:
                logger.info(f"   Tags: {', '.join(tags)}")
            if args.offline_mode:
                logger.info(f"   Offline mode: metrics saved locally, will sync when network is stable")
            sys.stdout.flush()
        
        # Не трогаем default upload destination: передача None иногда приводит к ошибкам вида
        # "Failed creating storage object file:// Reason: 'NoneType' object has no attribute 'startswith'"
            
    except Exception as e:
        logger.error(f"Failed to initialize ClearML task: {e}")
        logger.error("Possible solutions:")
        logger.error("  1. Check your network connection")
        logger.error("  2. Use --offline-mode flag to save metrics locally")
        logger.error("  3. Try again later when network is stable")
        sys.exit(1)
    
    logger.info("")
    logger.info("="*80)
    logger.info("Starting upload process...")
    logger.info("="*80)
    sys.stdout.flush()
    
    # Загружаем результаты
    overall_start_time = time.time()
    upload_summary = {
        'config': {'success': False, 'error': None},
        'checkpoints': {'success': False, 'error': None},
        'metrics': {'success': False, 'error': None},
        'visualizations': {'success': False, 'error': None},
        'logs': {'success': False, 'error': None}
    }
    
    try:
        # Загружаем конфигурацию
        if not args.skip_config:
            logger.info("")
            try:
                upload_config(task, exp_dir)
                upload_summary['config']['success'] = True
            except Exception as e:
                upload_summary['config']['error'] = str(e)
                logger.error(f"Failed to upload config: {e}")
                sys.stdout.flush()
        else:
            upload_summary['config']['success'] = None  # Skipped
        
        # Загружаем чекпоинты - ОТКЛЮЧЕНО
        upload_summary['checkpoints']['success'] = None  # Skipped
        
        # Загружаем метрики
        logger.info("")
        try:
            upload_metrics(
                task,
                exp_dir,
                skip_training_metrics=args.skip_training_metrics,
                batch_size=args.batch_size,
            )
            upload_summary['metrics']['success'] = True
        except Exception as e:
            upload_summary['metrics']['error'] = str(e)
            logger.error(f"Failed to upload metrics: {e}")
            sys.stdout.flush()
        
        # Загружаем визуализации - ОТКЛЮЧЕНО
        upload_summary['visualizations']['success'] = None  # Skipped
        
        # Загружаем логи
        if not args.skip_logs:
            logger.info("")
            try:
                upload_logs(task, exp_dir)
                upload_summary['logs']['success'] = True
            except Exception as e:
                upload_summary['logs']['error'] = str(e)
                logger.error(f"Failed to upload logs: {e}")
                sys.stdout.flush()
        else:
            upload_summary['logs']['success'] = None  # Skipped
        
        total_elapsed = time.time() - overall_start_time
        
        # Выводим итоговую сводку
        logger.info("")
        logger.info("="*80)
        logger.info("Upload Summary:")
        logger.info("="*80)
        
        for component, status in upload_summary.items():
            if status['success'] is None:
                logger.info(f"  {component.capitalize()}: SKIPPED")
            elif status['success']:
                logger.info(f"  {component.capitalize()}: SUCCESS")
            else:
                logger.info(f"  {component.capitalize()}: FAILED - {status['error']}")
        
        logger.info("")
        logger.info(f"Total time: {total_elapsed:.1f}s ({total_elapsed/60:.1f} minutes)")
        
        # Показываем URL только если хотя бы что-то загрузилось
        successful_uploads = sum(1 for s in upload_summary.values() if s['success'] is True)
        if successful_uploads > 0:
            try:
                logger.info(f"Task URL: {task.get_output_log_web_page()}")
            except Exception:
                pass
        
        logger.info("="*80)
        sys.stdout.flush()
        
    except KeyboardInterrupt:
        logger.warning("Upload interrupted by user (Ctrl+C)")
        logger.info("Partial upload summary:")
        for component, status in upload_summary.items():
            if status['success'] is True:
                logger.info(f"  {component.capitalize()}: SUCCESS")
            elif status['success'] is False:
                logger.info(f"  {component.capitalize()}: FAILED - {status['error']}")
    except Exception as e:
        logger.error(f"Unexpected error during upload: {e}")
        logger.info("Partial upload summary:")
        for component, status in upload_summary.items():
            if status['success'] is True:
                logger.info(f"  {component.capitalize()}: SUCCESS")
            elif status['success'] is False:
                logger.info(f"  {component.capitalize()}: FAILED - {status['error']}")
    finally:
        # Закрываем задачу
        try:
            task.close()
        except Exception:
            pass


if __name__ == '__main__':
    if not CLEARML_AVAILABLE:
        logger.error("ClearML is not available")
        sys.exit(1)
    main()
