#!/usr/bin/env python3
"""
Скрипт для сбора всех изображений из LFW датасета в одну папку
"""

import os
import shutil
from tqdm import tqdm
import sys

def collect_lfw_images(source_dir: str, output_dir: str, max_images: int = 1000):
    """
    Собирает все изображения из подпапок LFW в одну папку
    
    Args:
        source_dir: Папка с LFW датасетом (lfw_funneled)
        output_dir: Папка для сохранения всех изображений
        max_images: Максимальное количество изображений для сбора
    """
    print("=" * 60)
    print("СБОР ИЗОБРАЖЕНИЙ ИЗ LFW ДАТАСЕТА")
    print("=" * 60)
    
    # Создаем выходную папку
    os.makedirs(output_dir, exist_ok=True)
    
    # Получаем список всех подпапок
    subfolders = [f for f in os.listdir(source_dir) 
                 if os.path.isdir(os.path.join(source_dir, f))]
    
    print(f"📁 Найдено папок: {len(subfolders)}")
    print(f"🎯 Цель: собрать до {max_images} изображений")
    print()
    
    total_copied = 0
    skipped = 0
    
    # Обрабатываем каждую папку
    for folder in tqdm(subfolders, desc="Обработка папок"):
        folder_path = os.path.join(source_dir, folder)
        
        # Получаем все JPG файлы в папке
        image_files = [f for f in os.listdir(folder_path) 
                      if f.lower().endswith(('.jpg', '.jpeg'))]
        
        for image_file in image_files:
            if total_copied >= max_images:
                print(f"\n⚠️  Достигнут лимит в {max_images} изображений")
                break
            
            source_path = os.path.join(folder_path, image_file)
            
            # Создаем уникальное имя файла
            # Заменяем пробелы и другие символы
            safe_folder = folder.replace(" ", "_").replace("'", "")
            # Укорачиваем слишком длинные имена
            if len(safe_folder) > 30:
                safe_folder = safe_folder[:30]
            
            # Создаем новое имя файла
            new_filename = f"{safe_folder}_{image_file}"
            dest_path = os.path.join(output_dir, new_filename)
            
            # Если файл уже существует, добавляем номер
            counter = 1
            while os.path.exists(dest_path):
                name, ext = os.path.splitext(new_filename)
                dest_path = os.path.join(output_dir, f"{name}_{counter}{ext}")
                counter += 1
            
            try:
                # Копируем файл
                shutil.copy2(source_path, dest_path)
                total_copied += 1
                
            except Exception as e:
                skipped += 1
                print(f"⚠️  Ошибка копирования {source_path}: {e}")
        
        if total_copied >= max_images:
            break
    
    print("\n" + "=" * 60)
    print("📊 РЕЗУЛЬТАТЫ:")
    print(f"✅ Скопировано изображений: {total_copied}")
    print(f"⚠️  Пропущено: {skipped}")
    print(f"📁 Изображения сохранены в: {output_dir}")
    
    # Показываем примеры файлов
    print("\n📄 Примеры сохраненных файлов:")
    sample_files = os.listdir(output_dir)[:5]
    for file in sample_files:
        file_path = os.path.join(output_dir, file)
        size_kb = os.path.getsize(file_path) / 1024
        print(f"  • {file} ({size_kb:.1f} KB)")
    
    return total_copied

def create_image_list(output_dir: str):
    """
    Создает текстовый файл со списком всех изображений
    
    Args:
        output_dir: Папка с изображениями
    """
    list_file = os.path.join(output_dir, "image_list.txt")
    
    images = [f for f in os.listdir(output_dir) 
             if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    with open(list_file, 'w', encoding='utf-8') as f:
        f.write(f"Всего изображений: {len(images)}\n")
        f.write("=" * 40 + "\n\n")
        
        for image in sorted(images):
            # Извлекаем имя человека из названия файла
            parts = image.split('_')
            person = " ".join(parts[:-1])  # Все части кроме последней (номера файла)
            
            file_path = os.path.join(output_dir, image)
            size_kb = os.path.getsize(file_path) / 1024
            
            f.write(f"{image:<50} | {person:<30} | {size_kb:6.1f} KB\n")
    
    print(f"\n📝 Список файлов создан: {list_file}")

def main():
    """Главная функция"""
    # Определяем пути
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Папка с LFW датасетом
    lfw_funneled = os.path.join(base_dir, "lfw_dataset", "lfw_funneled")
    
    if not os.path.exists(lfw_funneled):
        print(f"❌ Папка LFW не найдена: {lfw_funneled}")
        print("\nСначала скачайте и распакуйте LFW датасет:")
        print("1. Запустите python download_lfw.py")
        print("2. Или скачайте вручную с Kaggle")
        return
    
    # Папка для сохранения всех изображений
    output_dir = os.path.join(base_dir, "lfw_dataset", "all_faces")
    
    # Сколько изображений собрать (рекомендуется 500-1000 для тестирования)
    max_images = 13000
    
    # Собираем изображения
    collected = collect_lfw_images(lfw_funneled, output_dir, max_images)
    
    if collected > 0:
        # Создаем список файлов
        create_image_list(output_dir)
        
        # Добавляем в датасет проекта
        add_to_project_dataset(output_dir)
    
    print("\n" + "=" * 60)
    print("✅ Готово! Теперь можно:")
    print("1. Использовать изображения для тестирования")
    print("2. Обучить модель на разнообразных лицах")
    print("=" * 60)

def add_to_project_dataset(source_dir: str, max_to_add: int = 13000):
    """
    Добавляет часть изображений в папку 'Unknown' проекта
    
    Args:
        source_dir: Папка с собранными изображениями
        max_to_add: Сколько изображений добавить в проект
    """
    print("\n➕ ДОБАВЛЕНИЕ В ПРОЕКТ:")
    
    # Папка 'Unknown' в проекте
    unknown_dir = os.path.join("dataset", "Unknown")
    os.makedirs(unknown_dir, exist_ok=True)
    
    # Получаем список изображений
    images = [f for f in os.listdir(source_dir) 
             if f.lower().endswith(('.jpg', '.jpeg'))]
    
    if not images:
        print("❌ Нет изображений для добавления")
        return
    
    # Ограничиваем количество
    images_to_add = images[:max_to_add]
    
    print(f"Добавляем {len(images_to_add)} изображений в {unknown_dir}")
    
    added = 0
    for image in tqdm(images_to_add, desc="Добавление в проект"):
        source_path = os.path.join(source_dir, image)
        dest_path = os.path.join(unknown_dir, f"lfw_{image}")
        
        # Если файл уже существует, пропускаем
        if os.path.exists(dest_path):
            continue
        
        try:
            shutil.copy2(source_path, dest_path)
            added += 1
        except Exception as e:
            print(f"⚠️  Ошибка копирования {image}: {e}")
    
    print(f"✅ Добавлено {added} новых изображений в датасет проекта")

if __name__ == "__main__":
    main()
