"""Утилиты для тестирования и визуализации моделей."""
import os
from typing import List, Optional
from mmengine.config import Config
from mmengine.runner import Runner


def setup_test_config(cfg: Config, data_root: str, split: str = 'test') -> Config:
    """Настраивает конфигурацию для тестирования.
    
    Args:
        cfg: Конфигурация MMSegmentation
        data_root: Корневая директория датасета
        split: Раздел датасета (train/val/test)
        
    Returns:
        Обновленная конфигурация
    """
    if data_root and hasattr(cfg, 'test_dataloader'):
        cfg.test_dataloader.dataset.data_root = data_root
        
        if 'data_prefix' in cfg.test_dataloader.dataset:
            # Определяем базовое имя для seg_map_path
            seg_map_base = 'labels_sam2'
            if hasattr(cfg, 'val_dataloader') and 'data_prefix' in cfg.val_dataloader.dataset:
                val_seg_path = cfg.val_dataloader.dataset.data_prefix.get('seg_map_path', '')
                if val_seg_path:
                    seg_map_base = val_seg_path.split('/')[0]
            elif 'seg_map_path' in cfg.test_dataloader.dataset.data_prefix:
                test_seg_path = cfg.test_dataloader.dataset.data_prefix.get('seg_map_path', '')
                if test_seg_path:
                    seg_map_base = test_seg_path.split('/')[0]
            
            # Устанавливаем пути в зависимости от split
            if split == 'train':
                cfg.test_dataloader.dataset.data_prefix['img_path'] = 'img/train'
                cfg.test_dataloader.dataset.data_prefix['seg_map_path'] = f'{seg_map_base}/train'
            elif split == 'val':
                cfg.test_dataloader.dataset.data_prefix['img_path'] = 'img/val'
                cfg.test_dataloader.dataset.data_prefix['seg_map_path'] = f'{seg_map_base}/val'
            elif split == 'test':
                cfg.test_dataloader.dataset.data_prefix['img_path'] = 'img/test'
                cfg.test_dataloader.dataset.data_prefix['seg_map_path'] = f'{seg_map_base}/test'
    
    return cfg


def get_class_names(cfg: Config) -> Optional[List[str]]:
    """Извлекает имена классов из конфигурации.
    
    Args:
        cfg: Конфигурация MMSegmentation
        
    Returns:
        Список имен классов или None
    """
    try:
        if hasattr(cfg, 'metainfo') and 'classes' in cfg.metainfo:
            return list(cfg.metainfo['classes'])
        elif hasattr(cfg, 'train_dataloader') and 'metainfo' in cfg.train_dataloader.dataset:
            return list(cfg.train_dataloader.dataset.metainfo.get('classes', []))
    except Exception:
        pass
    return None


def get_class_names_from_dataloader(dataloader) -> Optional[List[str]]:
    """Извлекает имена классов из dataloader.
    
    Args:
        dataloader: DataLoader MMSegmentation
        
    Returns:
        Список имен классов или None
    """
    try:
        if hasattr(dataloader, 'dataset') and hasattr(dataloader.dataset, 'metainfo'):
            if 'classes' in dataloader.dataset.metainfo:
                return list(dataloader.dataset.metainfo['classes'])
    except Exception:
        pass
    return None
