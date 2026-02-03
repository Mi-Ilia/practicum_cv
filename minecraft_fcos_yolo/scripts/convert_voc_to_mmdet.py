"""
Конвертация Pascal VOC XML аннотаций в формат MMDetection BaseDetDataset.

Скрипт конвертирует XML файлы в JSON формат, необходимый для обучения FCOS.
Результаты сохраняются в datasets/minecraft/annotations/.
"""

import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Tuple
from PIL import Image
from tqdm import tqdm


# Классы мобов Minecraft (должны соответствовать configs/fcos/fcos_minecraft.py)
CLASSES = (
    'bee', 'chicken', 'cow', 'creeper', 'enderman', 'fox', 'frog',
    'ghast', 'goat', 'llama', 'pig', 'sheep', 'skeleton', 'spider',
    'turtle', 'wolf', 'zombie'
)

# Создаем словарь для быстрого поиска индексов классов
CLASS_TO_ID = {cls: idx for idx, cls in enumerate(CLASSES)}


def parse_voc_xml(xml_path: Path, verbose: bool = False) -> Tuple[str, int, int, List[Dict]]:
    """
    Парсит Pascal VOC XML файл и возвращает информацию об изображении.
    
    Args:
        xml_path: Путь к XML файлу
        verbose: Выводить предупреждения
        
    Returns:
        Tuple (filename, width, height, instances)
        instances: список словарей с bbox и bbox_label
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    # Получаем имя файла
    filename_elem = root.find('filename')
    filename = filename_elem.text if filename_elem is not None else None
    if filename is None:
        filename = xml_path.stem + '.jpg'
    
    # Получаем размеры изображения
    size = root.find('size')
    width = int(size.find('width').text)
    height = int(size.find('height').text)
    
    # Получаем объекты
    instances = []
    for obj in root.findall('object'):
        name_elem = obj.find('name')
        if name_elem is None:
            if verbose:
                print(f"⚠ Предупреждение: объект без имени в {xml_path}, пропускаем")
            continue
            
        name = name_elem.text
        difficult = int(obj.find('difficult').text) if obj.find('difficult') is not None else 0
        
        # Пропускаем объекты, которых нет в списке классов
        if name not in CLASS_TO_ID:
            if verbose:
                print(f"⚠ Предупреждение: класс '{name}' не найден в списке классов (файл {xml_path.name}), пропускаем")
            continue
        
        # Получаем bbox
        bndbox = obj.find('bndbox')
        if bndbox is None:
            if verbose:
                print(f"⚠ Предупреждение: объект без bbox в {xml_path}, пропускаем")
            continue
            
        try:
            xmin = int(float(bndbox.find('xmin').text))
            ymin = int(float(bndbox.find('ymin').text))
            xmax = int(float(bndbox.find('xmax').text))
            ymax = int(float(bndbox.find('ymax').text))
        except (ValueError, AttributeError) as e:
            if verbose:
                print(f"⚠ Предупреждение: ошибка парсинга bbox в {xml_path}: {e}, пропускаем")
            continue
        
        # Проверяем валидность bbox
        if xmax <= xmin or ymax <= ymin:
            if verbose:
                print(f"⚠ Предупреждение: невалидный bbox [{xmin}, {ymin}, {xmax}, {ymax}] в {xml_path.name}, пропускаем")
            continue
        
        instances.append({
            'bbox': [xmin, ymin, xmax, ymax],  # Формат [x1, y1, x2, y2]
            'bbox_label': CLASS_TO_ID[name],
            'ignore_flag': difficult  # Помечаем difficult объекты как ignore
        })
    
    return filename, width, height, instances


def verify_image_size(img_path: Path, xml_width: int, xml_height: int, verbose: bool = True) -> Tuple[int, int]:
    """
    Проверяет и возвращает реальный размер изображения.
    Если размер не совпадает с XML, возвращает реальный размер.
    
    Args:
        img_path: Путь к изображению
        xml_width: Ширина из XML
        xml_height: Высота из XML
        verbose: Выводить предупреждения
        
    Returns:
        Tuple (width, height) - реальный размер изображения
    """
    if img_path.exists():
        try:
            with Image.open(img_path) as img:
                real_width, real_height = img.size
                if real_width != xml_width or real_height != xml_height:
                    if verbose:
                        print(f"⚠ Размер изображения {img_path.name} не совпадает с XML: "
                              f"XML={xml_width}x{xml_height}, реальный={real_width}x{real_height}")
                    return real_width, real_height
                return xml_width, xml_height
        except Exception as e:
            if verbose:
                print(f"⚠ Ошибка при открытии изображения {img_path}: {e}")
            return xml_width, xml_height
    return xml_width, xml_height


def convert_split(split_name: str, dataset_root: Path, annotations_dir: Path, verbose: bool = True) -> Dict[str, int]:
    """
    Конвертирует все XML файлы из указанного сплита в JSON формат.
    
    Args:
        split_name: Имя сплита ('train', 'valid', 'test')
        dataset_root: Корневая директория датасета
        annotations_dir: Директория для сохранения JSON файлов
        verbose: Выводить предупреждения в процессе обработки
        
    Returns:
        Словарь со статистикой обработки
    """
    split_dir = dataset_root / split_name
    if not split_dir.exists():
        if verbose:
            print(f"⚠ Директория {split_dir} не найдена, пропускаем")
        return {'skipped': 1, 'reason': 'directory_not_found'}
    
    # Получаем все XML файлы
    xml_files = sorted(list(split_dir.glob('*.xml')))
    if not xml_files:
        if verbose:
            print(f"⚠ XML файлы не найдены в {split_dir}")
        return {'skipped': 1, 'reason': 'no_xml_files'}
    
    print(f"\n{'='*70}")
    print(f"Конвертация сплита: {split_name}")
    print(f"{'='*70}")
    print(f"Найдено XML файлов: {len(xml_files)}")
    
    # Создаем структуру данных
    data_list = []
    img_id_counter = 0
    
    # Статистика обработки
    stats = {
        'total_xml': len(xml_files),
        'success': 0,
        'skipped_no_image': 0,
        'skipped_no_objects': 0,
        'skipped_invalid_bbox': 0,
        'skipped_unknown_class': 0,
        'skipped_errors': 0,
        'size_mismatches': 0
    }
    
    # Категории для CocoMetric (требуется для evaluation)
    # ID категорий 0-based для согласованности с bbox_label
    categories = [
        {'id': idx, 'name': cls} for idx, cls in enumerate(CLASSES)
    ]
    
    # Обрабатываем каждый XML файл
    for xml_path in tqdm(xml_files, desc=f"Обработка {split_name}"):
        try:
            # Парсим XML
            filename, xml_width, xml_height, instances = parse_voc_xml(xml_path, verbose=verbose)
            
            # Учитываем пропущенные классы из parse_voc_xml
            # (это делается внутри функции, поэтому статистика приблизительная)
            
            # Проверяем изображение
            img_path = split_dir / filename
            if not img_path.exists():
                # Пробуем найти изображение с другим расширением
                possible_extensions = ['.jpg', '.jpeg', '.png']
                img_path = None
                for ext in possible_extensions:
                    candidate = split_dir / (xml_path.stem + ext)
                    if candidate.exists():
                        img_path = candidate
                        filename = candidate.name
                        break
                
                if img_path is None:
                    if verbose:
                        print(f"⚠ Изображение не найдено для {xml_path.name}, пропускаем")
                    stats['skipped_no_image'] += 1
                    continue
            
            # Проверяем размер изображения
            width, height = verify_image_size(img_path, xml_width, xml_height, verbose=verbose)
            if width != xml_width or height != xml_height:
                stats['size_mismatches'] += 1
            
            # Пропускаем изображения без объектов (для FCOS это нормально)
            if not instances:
                stats['skipped_no_objects'] += 1
                continue
            
            # Создаем запись для data_list
            # MMDetection 3.x требует оба ключа: 'img' для data_prefix и 'img_path' для пайплайна
            # Путь относительно data_root: split_name/filename
            relative_path = f"{split_name}/{filename}"
            data_entry = {
                'img': relative_path,       # Для data_prefix (обязательно!)
                'img_path': relative_path,  # Для LoadImageFromFile
                'height': height,
                'width': width,
                'img_id': img_id_counter,  # Уникальный ID для CocoMetric
                'instances': instances
            }
            
            data_list.append(data_entry)
            img_id_counter += 1
            stats['success'] += 1
            
        except Exception as e:
            if verbose:
                print(f"✗ Ошибка при обработке {xml_path}: {e}")
            stats['skipped_errors'] += 1
            continue
    
    # Создаем финальную структуру JSON
    result = {
        'metainfo': {
            'classes': CLASSES,
        },
        'data_list': data_list,
        'categories': categories  # Для CocoMetric
    }
    
    # Сохраняем JSON файл
    annotations_dir.mkdir(parents=True, exist_ok=True)
    
    # Определяем имя выходного файла
    if split_name == 'valid':
        output_filename = 'val_annotations.voc.json'
    else:
        output_filename = f'{split_name}_annotations.voc.json'
    
    output_path = annotations_dir / output_filename
    
    # Выводим статистику
    print(f"\n✓ Успешно обработано: {stats['success']}")
    print(f"✓ Сохранено в: {output_path}")
    
    if stats['skipped_no_image'] > 0:
        print(f"  ⚠ Пропущено (нет изображения): {stats['skipped_no_image']}")
    if stats['skipped_no_objects'] > 0:
        print(f"  ⚠ Пропущено (нет объектов): {stats['skipped_no_objects']}")
    if stats['skipped_errors'] > 0:
        print(f"  ✗ Ошибок при обработке: {stats['skipped_errors']}")
    if stats['size_mismatches'] > 0:
        print(f"  ⚠ Несоответствий размеров XML/изображение: {stats['size_mismatches']}")
    
    # Сохраняем с правильным форматированием
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Конвертация {split_name} завершена")
    
    return stats


def rename_valid_to_val(dataset_root: Path) -> None:
    """
    Переименовывает папку 'valid' в 'val' для соответствия стандарту MMDetection.
    
    Args:
        dataset_root: Корневая директория датасета
    """
    valid_path = dataset_root / 'valid'
    val_path = dataset_root / 'val'
    
    if valid_path.exists() and not val_path.exists():
        print(f"\n{'='*70}")
        print("ПЕРЕИМЕНОВАНИЕ: valid → val")
        print(f"{'='*70}")
        print(f"Переименование: {valid_path.name} → {val_path.name}")
        try:
            valid_path.rename(val_path)
            print(f"✓ Папка успешно переименована")
        except Exception as e:
            print(f"✗ Ошибка при переименовании: {e}")
            return
    elif valid_path.exists() and val_path.exists():
        print(f"\n⚠ Папки 'valid' и 'val' существуют одновременно")
        print(f"   Используется папка 'val' (стандарт MMDetection)")


def main():
    """Основная функция."""
    print("=" * 70)
    print("КОНВЕРТАЦИЯ PASCAL VOC XML → MMDetection BaseDetDataset JSON")
    print("=" * 70)
    
    # Пути
    dataset_root = Path('datasets/minecraft')
    annotations_dir = dataset_root / 'annotations'
    
    # Переименовываем valid → val перед обработкой (для соответствия стандарту MMDetection)
    rename_valid_to_val(dataset_root)
    
    # Проверяем структуру датасета
    print(f"\nПроверка структуры датасета:")
    print(f"  Корневая директория: {dataset_root.absolute()}")
    
    # Используем 'val' вместо 'valid' после переименования
    splits_to_check = ['train', 'val', 'test']
    for split in splits_to_check:
        split_path = dataset_root / split
        if split_path.exists():
            xml_count = len(list(split_path.glob('*.xml')))
            jpg_count = len(list(split_path.glob('*.jpg')))
            print(f"  {split}/: {xml_count} XML, {jpg_count} JPG")
        else:
            print(f"  {split}/: ⚠ не найдено")
    
    # Конвертируем каждый сплит и собираем статистику
    # Важно: convert_split принимает имя папки, но для 'val' использует правильное имя файла
    all_stats = {}
    for split in splits_to_check:
        split_path = dataset_root / split
        if split_path.exists():
            stats = convert_split(split, dataset_root, annotations_dir, verbose=True)
            if stats:
                all_stats[split] = stats
    
    # Итоговая статистика
    print("\n" + "=" * 70)
    print("ИТОГОВАЯ СТАТИСТИКА")
    print("=" * 70)
    
    total_success = 0
    total_skipped = 0
    
    for split, stats in all_stats.items():
        if isinstance(stats, dict) and 'success' in stats:
            success = stats['success']
            skipped = stats['total_xml'] - success
            total_success += success
            total_skipped += skipped
            print(f"\n{split.upper()}:")
            print(f"  Успешно: {success}")
            print(f"  Пропущено: {skipped}")
    
    print(f"\n{'='*70}")
    print(f"ВСЕГО:")
    print(f"  Успешно конвертировано: {total_success}")
    print(f"  Пропущено: {total_skipped}")
    print(f"{'='*70}")
    
    print("\n" + "=" * 70)
    print("КОНВЕРТАЦИЯ ЗАВЕРШЕНА")
    print("=" * 70)
    print(f"\nJSON файлы сохранены в: {annotations_dir.absolute()}")
    print("\nСозданные файлы:")
    for json_file in sorted(annotations_dir.glob('*.json')):
        file_size_kb = json_file.stat().st_size / 1024
        print(f"  - {json_file.name} ({file_size_kb:.1f} KB)")


if __name__ == '__main__':
    main()

