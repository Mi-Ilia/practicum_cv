"""Конвертация Pascal VOC XML аннотаций в YOLO формат (txt файлы).
Создает новый датасет в datasets/minecraft_yolo со структурой YOLO.
"""
import xml.etree.ElementTree as ET
from pathlib import Path
import shutil

# Классы Minecraft мобов (в том же порядке, что и в dataset.yaml)
classes = [
    "bee", "chicken", "cow", "creeper", "enderman",
    "fox", "frog", "ghast", "goat", "llama",
    "pig", "sheep", "skeleton", "spider", "turtle",
    "wolf", "zombie"
]

class_to_id = {cls: idx for idx, cls in enumerate(classes)}

# Пути
source_dataset = Path('datasets/minecraft')
target_dataset = Path('datasets/minecraft_yolo')

def convert_bbox_to_yolo(bbox, img_width, img_height):
    """Конвертирует bbox из VOC формата (x_min, y_min, x_max, y_max) в YOLO формат (x_center, y_center, width, height) - нормализованные."""
    x_min, y_min, x_max, y_max = bbox
    
    # Валидация bbox
    if x_max <= x_min or y_max <= y_min:
        return None  # Невалидный bbox
    
    # Ограничиваем координаты границами изображения
    x_min = max(0, min(x_min, img_width))
    y_min = max(0, min(y_min, img_height))
    x_max = max(0, min(x_max, img_width))
    y_max = max(0, min(y_max, img_height))
    
    # Проверяем размеры после ограничения
    if x_max <= x_min or y_max <= y_min:
        return None
    
    # Нормализуем координаты
    x_center = (x_min + x_max) / 2.0 / img_width
    y_center = (y_min + y_max) / 2.0 / img_height
    width = (x_max - x_min) / img_width
    height = (y_max - y_min) / img_height
    
    # Проверяем, что нормализованные координаты в допустимом диапазоне [0, 1]
    if not (0 <= x_center <= 1 and 0 <= y_center <= 1 and 0 < width <= 1 and 0 < height <= 1):
        return None
    
    return x_center, y_center, width, height

def parse_xml_annotation(xml_path):
    """Парсит XML файл и возвращает список аннотаций."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    # Получаем размеры изображения
    size = root.find('size')
    if size is None:
        return []
    
    width_elem = size.find('width')
    height_elem = size.find('height')
    if width_elem is None or height_elem is None:
        return []
    
    try:
        img_width = int(width_elem.text)
        img_height = int(height_elem.text)
    except (ValueError, AttributeError):
        return []
    
    # Проверяем валидность размеров
    if img_width <= 0 or img_height <= 0:
        return []
    
    annotations = []
    for obj in root.findall('object'):
        # Пропускаем difficult объекты
        difficult_elem = obj.find('difficult')
        if difficult_elem is not None and int(difficult_elem.text) == 1:
            continue
        
        name_elem = obj.find('name')
        if name_elem is None:
            continue
        
        class_name = name_elem.text
        if class_name not in class_to_id:
            continue
        
        bbox = obj.find('bndbox')
        if bbox is None:
            continue
        
        try:
            x_min = float(bbox.find('xmin').text)
            y_min = float(bbox.find('ymin').text)
            x_max = float(bbox.find('xmax').text)
            y_max = float(bbox.find('ymax').text)
        except (ValueError, AttributeError):
            continue
        
        class_id = class_to_id[class_name]
        yolo_bbox = convert_bbox_to_yolo([x_min, y_min, x_max, y_max], img_width, img_height)
        
        # Пропускаем невалидные bbox
        if yolo_bbox is None:
            continue
        
        annotations.append((class_id, yolo_bbox))
    
    return annotations

def convert_split(split_name):
    """Конвертирует все XML файлы в split в YOLO формат и копирует изображения."""
    source_split_dir = source_dataset / split_name
    target_split_dir = target_dataset / split_name
    
    if not source_split_dir.exists():
        print(f"! Папка {source_split_dir} не найдена")
        return 0, 0
    
    # Создаем целевую папку
    target_split_dir.mkdir(parents=True, exist_ok=True)
    
    xml_files = list(source_split_dir.glob('*.xml'))
    converted = 0
    skipped = 0
    skipped_no_annotations = 0
    skipped_no_image = 0
    images_copied = 0
    
    for xml_file in xml_files:
        try:
            annotations = parse_xml_annotation(xml_file)
            
            if not annotations:
                skipped_no_annotations += 1
                skipped += 1
                continue
            
            # Получаем имя изображения (без расширения)
            img_name_base = xml_file.stem
            
            # Ищем соответствующий файл изображения
            img_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
            source_img_file = None
            for ext in img_extensions:
                potential_img = source_split_dir / f"{img_name_base}{ext}"
                if potential_img.exists():
                    source_img_file = potential_img
                    break
            
            if source_img_file is None:
                # Ищем любой файл с таким именем (может быть другое расширение)
                img_files = list(source_split_dir.glob(f"{img_name_base}.*"))
                img_files = [f for f in img_files if f.suffix.lower() not in ['.xml', '.txt']]
                if img_files:
                    source_img_file = img_files[0]
            
            if source_img_file is None:
                skipped_no_image += 1
                skipped += 1
                continue
            
            # Копируем изображение в целевую папку
            target_img_file = target_split_dir / source_img_file.name
            shutil.copy2(source_img_file, target_img_file)
            images_copied += 1
            
            # Создаем txt файл с аннотациями YOLO формата
            txt_file = target_split_dir / f"{img_name_base}.txt"
            with open(txt_file, 'w') as f:
                for class_id, (x_center, y_center, width, height) in annotations:
                    f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
            
            converted += 1
            
        except Exception as e:
            print(f"Ошибка при обработке {xml_file.name}: {e}")
            skipped += 1
    
    print(f"[OK] {split_name}: конвертировано {converted} файлов, скопировано {images_copied} изображений")
    if skipped > 0:
        print(f"  ⚠ Пропущено: {skipped} (нет аннотаций: {skipped_no_annotations}, нет изображения: {skipped_no_image})")
    return converted, images_copied

# Конвертируем все splits
print("=" * 70)
print("СОЗДАНИЕ ДАТАСЕТА В ФОРМАТЕ YOLO")
print("=" * 70)

# Создаем корневую папку датасета
target_dataset.mkdir(parents=True, exist_ok=True)

total_annotations = 0
total_images = 0

for split in ['train', 'val', 'test']:
    ann_count, img_count = convert_split(split)
    total_annotations += ann_count
    total_images += img_count

# Создаем dataset.yaml файл
yaml_content = f"""# Minecraft Mob Detection Dataset for YOLO
path: {target_dataset.absolute()}  # dataset root dir
train: train  # train images (relative to 'path')
val: val  # val images (relative to 'path')
test: test  # test images (optional, relative to 'path')

# Classes
names:
  0: bee
  1: chicken
  2: cow
  3: creeper
  4: enderman
  5: fox
  6: frog
  7: ghast
  8: goat
  9: llama
  10: pig
  11: sheep
  12: skeleton
  13: spider
  14: turtle
  15: wolf
  16: zombie

nc: 17  # number of classes
"""

yaml_file = target_dataset / 'dataset.yaml'
with open(yaml_file, 'w', encoding='utf-8') as f:
    f.write(yaml_content)

print("\n" + "=" * 70)
print(f"[OK] Датасет создан в: {target_dataset}")
print(f"  - Аннотаций конвертировано: {total_annotations}")
print(f"  - Изображений скопировано: {total_images}")
print(f"  - dataset.yaml создан: {yaml_file}")
print("=" * 70)

