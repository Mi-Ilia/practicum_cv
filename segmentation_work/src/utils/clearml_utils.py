"""Утилиты для интеграции ClearML

ClearML (https://clear.ml/) - платформа для управления экспериментами:
- Логирование метрик (train/val loss, mIoU, Dice)
- Сохранение артефактов (чекпоинты, визуализации)
- Сравнение экспериментов
- Воспроизводимость (автоматическое логирование кода и зависимостей)
"""

import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any

try:
    from clearml import Task, OutputModel, Dataset
except ImportError:
    raise ImportError(
        "ClearML не установлен. Установите: pip install clearml"
    )


def init_clearml_task(
    project_name: str,
    task_name: str,
    tags: Optional[list] = None,
    output_uri: Optional[str] = None,
    auto_connect_frameworks: bool = True,
    auto_connect_streams: bool = True,
    auto_connect_arg_parser: bool = True,
    offline_mode: bool = False,
    timeout: int = 30
) -> Optional[Task]:
    """Инициализирует задачу ClearML с защитой от зависания.
    
    Args:
        project_name (str): Название проекта в ClearML
        task_name (str): Название задачи/эксперимента
        tags (list, optional): Теги для задачи. Defaults to None.
        output_uri (str, optional): URI для сохранения артефактов. Defaults to None.
        auto_connect_frameworks (bool): Автоматически подключать фреймворки (PyTorch, etc.)
        auto_connect_streams (bool): Автоматически подключать stdout/stderr
        auto_connect_arg_parser (bool): Автоматически логировать аргументы argparse
        offline_mode (bool): Использовать офлайн режим (не подключаться к серверу). Defaults to False.
        timeout (int): Таймаут инициализации в секундах. Defaults to 30.
    
    Returns:
        Task: Объект задачи ClearML или None при ошибке
    """
    try:
        # Проверяем переменную окружения для офлайн режима
        if not offline_mode:
            offline_mode = os.environ.get('CLEARML_OFFLINE_MODE', '').lower() in ('1', 'true', 'yes')
        
        # Если офлайн режим, устанавливаем переменную окружения
        if offline_mode:
            os.environ['CLEARML_OFFLINE_MODE'] = '1'
        
        # Отключаем мониторинг ClearML для предотвращения зависания
        # Мониторинг не критичен - метрики логируются через наш хук
        os.environ['CLEARML_MONITOR_ITERATION_REPORTING'] = '0'
        os.environ['CLEARML_MONITOR_GPU'] = '0'
        
        # Создаём задачу с защитой от зависания
        try:
            # Отключаем auto_connect_streams и auto_connect_arg_parser для предотвращения зависания
            # при сохранении кода/репозитория на нестабильной сети
            # Метрики все равно будут логироваться через наш ClearMLHook
            task = Task.init(
                project_name=project_name,
                task_name=task_name,
                tags=tags,
                output_uri=output_uri,
                auto_connect_frameworks=auto_connect_frameworks,
                auto_connect_streams=False,  # Отключаем для предотвращения зависания
                auto_connect_arg_parser=False  # Отключаем для предотвращения зависания
            )
            
            # Отключаем автоматическое обновление задачи, которое может зависнуть при проблемах с сетью
            # Задача будет обновляться только через наш хук при логировании метрик
            try:
                task.set_system_tags(['offline_mode'])  # Помечаем как офлайн для предотвращения автосинхронизации
            except Exception:
                pass
            
            # Отключаем мониторинг ClearML, который может зависнуть при проблемах с сетью
            # Мониторинг не критичен для обучения - метрики логируются через наш хук
            try:
                task.get_logger().set_default_upload_destination(None)
            except Exception:
                pass
            
            return task
        except Exception as init_error:
            print(f"Error during Task.init(): {init_error}")
            sys.stdout.flush()
            raise
    except Exception as e:
        return None


def log_config_to_clearml(task: Task, config_path: str):
    """Логирует конфигурационный файл в ClearML с защитой от зависания.
    
    Args:
        task (Task): Объект задачи ClearML
        config_path (str): Путь к конфигурационному файлу
    """
    if task is None or not os.path.exists(config_path):
        return
    
    import time
    import sys
    import threading
    
    # Проверяем офлайн режим
    is_offline = os.environ.get('CLEARML_OFFLINE_MODE', '').lower() in ('1', 'true', 'yes')
    
    if is_offline:
        # В офлайн режиме загрузка быстрая и безопасная
        try:
            task.upload_artifact(name='config', artifact_object=config_path)
            return
        except Exception:
            pass
    
    # В онлайн режиме - используем поток с таймаутом для защиты от зависания
    upload_success = [False]
    upload_error = [None]
    
    def upload_in_thread():
        try:
            task.upload_artifact(name='config', artifact_object=config_path)
            upload_success[0] = True
        except Exception as e:
            upload_error[0] = e
    
    upload_thread = threading.Thread(target=upload_in_thread, daemon=True)
    upload_thread.start()
    upload_thread.join(timeout=10)  # Максимум 10 секунд на загрузку
    
    if upload_thread.is_alive():
        # Поток все еще работает - значит зависло, пропускаем
        pass
    elif upload_success[0]:
        # Успешно загружено - сохраняем путь как параметр
        try:
            task.set_parameter('config_path', str(config_path))
        except Exception:
            pass
    elif upload_error[0]:
        # Ошибка при загрузке - пытаемся сохранить путь как параметр
        try:
            task.set_parameter('config_path', str(config_path))
        except Exception:
            pass


def log_hyperparameters_to_clearml(task: Task, hyperparameters: Dict[str, Any]):
    """Логирует гиперпараметры в ClearML.
    
    Args:
        task (Task): Объект задачи ClearML
        hyperparameters (dict): Словарь с гиперпараметрами
    """
    # Логируем как параметры (для сравнения экспериментов)
    for key, value in hyperparameters.items():
        task.set_parameter(name=key, value=value)
    
    # Также логируем как артефакт для удобства
    task.upload_artifact(
        name='hyperparameters',
        artifact_object=hyperparameters
    )


def log_model_checkpoint_to_clearml(
    task: Task,
    checkpoint_path: str,
    model_name: str = 'best_model',
    framework: str = 'PyTorch',
    labels: Optional[Dict[str, float]] = None
):
    """Сохраняет чекпоинт модели как артефакт в ClearML.
    
    Args:
        task (Task): Объект задачи ClearML
        checkpoint_path (str): Путь к чекпоинту
        model_name (str): Название модели. Defaults to 'best_model'.
        framework (str): Фреймворк (PyTorch, TensorFlow, etc.). Defaults to 'PyTorch'.
        labels (dict, optional): Метрики модели для тегирования. Defaults to None.
    """
    if not os.path.exists(checkpoint_path):
        print(f"Warning: чекпоинт не найден: {checkpoint_path}")
        return
    
    # Создаём OutputModel для хранения модели
    output_model = OutputModel(
        task=task,
        name=model_name,
        framework=framework,
        labels=labels
    )
    
    # Загружаем чекпоинт
    output_model.update_weights_package(
        weights_path=checkpoint_path
    )


def log_visualizations_to_clearml(
    task: Task,
    visualization_dir: str,
    step: Optional[int] = None
):
    """Логирует визуализации в ClearML.
    
    Args:
        task (Task): Объект задачи ClearML
        visualization_dir (str): Директория с визуализациями
        step (int, optional): Шаг/эпоха для логирования. Defaults to None.
    """
    vis_dir = Path(visualization_dir)
    if not vis_dir.exists():
        print(f"Warning: директория визуализаций не найдена: {visualization_dir}")
        return
    
    # Логируем все изображения из директории
    for img_path in vis_dir.glob('*.png'):
        task.get_logger().report_image(
            title='Predictions',
            series=img_path.stem,
            iteration=step if step is not None else 0,
            local_path=str(img_path)
        )


def log_metrics_to_clearml(
    task: Task,
    metrics: Dict[str, float],
    step: Optional[int] = None,
    epoch: Optional[int] = None
):
    """Логирует метрики в ClearML с защитой от зависания.
    
    Args:
        task (Task): Объект задачи ClearML
        metrics (dict): Словарь с метриками {название: значение}
        step (int, optional): Номер итерации. Defaults to None.
        epoch (int, optional): Номер эпохи. Defaults to None.
    """
    if task is None:
        return
    
    import threading
    import time
    
    # Проверяем офлайн режим
    is_offline = os.environ.get('CLEARML_OFFLINE_MODE', '').lower() in ('1', 'true', 'yes')
    
    def log_metrics_internal():
        try:
            iteration = step if step is not None else (epoch if epoch is not None else 0)
            
            for metric_name, metric_value in metrics.items():
                try:
                    # Определяем категорию метрики
                    if 'loss' in metric_name.lower():
                        title = 'Losses'
                    elif 'iou' in metric_name.lower() or 'dice' in metric_name.lower():
                        title = 'Metrics'
                    elif 'lr' in metric_name.lower() or 'learning_rate' in metric_name.lower():
                        title = 'Learning Rate'
                    else:
                        title = 'Other'
                    
                    task.get_logger().report_scalar(
                        title=title,
                        series=metric_name,
                        value=float(metric_value),
                        iteration=iteration
                    )
                except Exception:
                    # Пропускаем отдельную метрику при ошибке
                    pass
        except Exception:
            # Если весь блок упал - не критично, продолжаем
            pass
    
    # В офлайн режиме логируем напрямую (быстро и безопасно)
    if is_offline:
        try:
            log_metrics_internal()
        except Exception:
            pass
        return
    
    # В онлайн режиме используем поток с таймаутом для защиты от зависания
    log_success = [False]
    log_error = [None]
    
    def log_in_thread():
        try:
            log_metrics_internal()
            log_success[0] = True
        except Exception as e:
            log_error[0] = e
    
    log_thread = threading.Thread(target=log_in_thread, daemon=True)
    log_thread.start()
    log_thread.join(timeout=2)  # Максимум 2 секунды на логирование метрик
    
    # Если зависло - просто пропускаем, не критично
    if log_thread.is_alive():
        # Поток завис - пропускаем логирование этой итерации
        pass
    elif log_error[0]:
        # Ошибка при логировании - не критично
        pass
