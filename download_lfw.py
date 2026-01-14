#!/usr/bin/env python3
"""
Скрипт для скачивания и добавления LFW датасета в проект
"""

import sys
import os
import tarfile
import shutil
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import kagglehub
from src.dataset_utils import DatasetManager

def extract_tar_gz(tar_path, extract_to):
    """Распаковка .tgz архива"""
    print(f"📦 Распаковка {tar_path}...")
    try:
        with tarfile.open(tar_path, 'r:gz') as tar:
            # Считаем общее количество файлов для прогресс-бара
            members = tar.getmembers()
            total = len(members)
            print(f"  Найдено {total} файлов в архиве")
            
            # Распаковываем
            for i, member in enumerate(members, 1):
                tar.extract(member, path=extract_to)
                if i % 1000 == 0:
                    print(f"  Распаковано {i}/{total} файлов")
            
        print(f"✅ Архив распакован в {extract_to}")
        return True
    except Exception as e:
        print(f"❌ Ошибка распаковки: {e}")
        return False

def main():
    print("=" * 60)
    print("Скачивание LFW (Labeled Faces in the Wild) датасета")
    print("=" * 60)
    
    # 1. Скачиваем датасет
    print("\n1. Скачивание датасета с Kaggle...")
    try:
        path = kagglehub.dataset_download("atulanandjha/lfwpeople")
        print(f"✅ Датасет скачан в: {path}")
    except Exception as e:
        print(f"❌ Ошибка скачивания: {e}")
        print("\nАльтернативные способы:")
        print("1. Скачайте вручную с: https://www.kaggle.com/datasets/atulanandjha/lfwpeople")
        print("2. Распакуйте в папку 'lfw_dataset' в корне проекта")
        return
    
    # 2. Ищем архив lfw-funneled.tgz
    print("\n2. Поиск архива с изображениями...")
    
    tar_file = None
    for file in os.listdir(path):
        if file == "lfw-funneled.tgz":
            tar_file = os.path.join(path, file)
            break
    
    if not tar_file:
        print("❌ Архив lfw-funneled.tgz не найден")
        print("\nСодержимое скачанной папки:")
        for item in os.listdir(path):
            item_path = os.path.join(path, item)
            if os.path.isdir(item_path):
                print(f"📁 {item} ({len(os.listdir(item_path))} элементов)")
            else:
                size = os.path.getsize(item_path) / (1024*1024)
                print(f"📄 {item} ({size:.1f} MB)")
        return
    
    print(f"✅ Найден архив: {tar_file}")
    size_mb = os.path.getsize(tar_file) / (1024*1024)
    print(f"   Размер: {size_mb:.1f} MB")
    
    # 3. Распаковываем архив
    print("\n3. Распаковка архива...")
    
    # Создаем папку для распаковки
    extract_dir = os.path.join(os.path.dirname(__file__), "lfw_dataset")
    os.makedirs(extract_dir, exist_ok=True)
    
    if extract_tar_gz(tar_file, extract_dir):
        # 4. Ищем распакованную папку lfw_funneled
        print("\n4. Поиск распакованной папки...")
        
        lfw_folder = None
        for root, dirs, files in os.walk(extract_dir):
            if "lfw_funneled" in dirs:
                lfw_folder = os.path.join(root, "lfw_funneled")
                break
        
        if lfw_folder and os.path.exists(lfw_folder):
            print(f"✅ Найдена папка с изображениями: {lfw_folder}")
            
            # Подсчитываем количество людей
            person_count = len([d for d in os.listdir(lfw_folder) 
                              if os.path.isdir(os.path.join(lfw_folder, d))])
            print(f"   Людей в датасете: {person_count}")
            
            # Подсчитываем общее количество фото
            total_photos = 0
            for person in os.listdir(lfw_folder)[:5]:  # Проверяем первые 5
                person_path = os.path.join(lfw_folder, person)
                if os.path.isdir(person_path):
                    photos = len([f for f in os.listdir(person_path) 
                                if f.lower().endswith('.jpg')])
                    total_photos += photos
                    if person_count <= 5:  # Показываем только если людей мало
                        print(f"   {person}: {photos} фото")
            
            if person_count > 5:
                print(f"   ... и еще {person_count-5} человек")
            
        else:
            print("❌ Папка lfw_funneled не найдена после распаковки")
            print("Содержимое распакованной папки:")
            for item in os.listdir(extract_dir)[:10]:
                print(f"  {item}")
            return
    else:
        return
    
    # 5. Добавляем датасет в проект
    print("\n5. Добавление лиц в папку 'Unknown'...")
    try:
        manager = DatasetManager()
        # Добавляем только 10 фото от каждого человека (чтобы не перегружать)
        added = manager.add_lfw_dataset(lfw_folder, max_per_person=10)
        print(f"✅ Добавлено примерно {added} фото из LFW датасета!")
    except Exception as e:
        print(f"❌ Ошибка добавления: {e}")
        import traceback
        traceback.print_exc()
    
    # 6. Показываем статистику
    print("\n6. Обновление статистики...")
    try:
        stats = manager.get_dataset_stats()
        print("\nТекущая статистика датасета:")
        print("-" * 40)
        for person, count in stats.items():
            print(f"{person:20} : {count:6} фото")
        print("-" * 40)
        
        total = sum(stats.values())
        print(f"Всего фото: {total}")
        
        # Проверяем наличие фото Aleksanderа
        if stats.get("Aleksander", 0) < 10:
            print("\n⚠️  ВНИМАНИЕ: У Aleksanderа меньше 10 фото!")
            print("   Захватите фото через GUI или скрипт capture_photos.py")
        
    except Exception as e:
        print(f"Ошибка получения статистики: {e}")
    
    print("\n" + "=" * 60)
    print("Датасет готов! Теперь можно обучить модель:")
    print("1. Запустите main.py")
    print("2. Нажмите 'Обновить модель' в GUI")
    print("=" * 60)
    
    # Сохраняем путь к датасету для будущего использования
    dataset_info = {
        "lfw_path": lfw_folder,
        "extracted_dir": extract_dir,
        "archive_path": tar_file,
        "downloaded_at": os.path.getmtime(tar_file)
    }
    
    info_file = os.path.join(extract_dir, "dataset_info.json")
    import json
    with open(info_file, 'w') as f:
        json.dump(dataset_info, f, indent=2)
    
    print(f"\n📁 Информация о датасете сохранена в: {info_file}")

if __name__ == "__main__":
    main()
