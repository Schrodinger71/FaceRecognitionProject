#!/usr/bin/env python3
"""
Скрипт для скачивания и добавления LFW датасета в проект
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import kagglehub
from src.dataset_utils import DatasetManager

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
    
    # 2. Находим путь к папке lfw-funneled
    print("\n2. Поиск папки с изображениями...")
    
    # Ищем папку lfw_funneled или lfw-funneled
    lfw_folder = None
    for root, dirs, files in os.walk(path):
        if "lfw_funneled" in dirs:
            lfw_folder = os.path.join(root, "lfw_funneled")
            break
        elif "lfw-funneled" in dirs:
            lfw_folder = os.path.join(root, "lfw-funneled")
            break
    
    if lfw_folder and os.path.exists(lfw_folder):
        print(f"✅ Найдена папка с изображениями: {lfw_folder}")
    else:
        # Если папка не найдена, проверяем корневую структуру
        possible_paths = [
            os.path.join(path, "lfw_funneled"),
            os.path.join(path, "lfw-funneled"),
            path  # возможно, уже в нужной папке
        ]
        
        for p in possible_paths:
            if os.path.exists(p) and len(os.listdir(p)) > 100:
                lfw_folder = p
                print(f"✅ Используем папку: {lfw_folder}")
                break
    
    if not lfw_folder:
        print("❌ Не удалось найти папку с изображениями")
        print("\nСтруктура скачанного датасета:")
        for item in os.listdir(path)[:10]:
            item_path = os.path.join(path, item)
            if os.path.isdir(item_path):
                print(f"📁 {item} ({len(os.listdir(item_path))} элементов)")
            else:
                print(f"📄 {item}")
        return
    
    # 3. Добавляем датасет в проект
    print("\n3. Добавление лиц в папку 'Неизвестный'...")
    try:
        manager = DatasetManager()
        manager.add_lfw_dataset(lfw_folder, max_per_person=20)
        print("✅ LFW датасет успешно добавлен!")
    except Exception as e:
        print(f"❌ Ошибка добавления: {e}")
    
    # 4. Показываем статистику
    print("\n4. Обновление статистики...")
    stats = manager.get_dataset_stats()
    print("\nТекущая статистика датасета:")
    print("-" * 30)
    for person, count in stats.items():
        print(f"{person:15} : {count:4} фото")
    print("-" * 30)
    
    total = sum(stats.values())
    print(f"Всего фото: {total}")
    
    print("\n" + "=" * 60)
    print("Датасет готов! Теперь можно обучить модель:")
    print("1. Запустите main.py")
    print("2. Нажмите 'Обновить модель' в GUI")
    print("=" * 60)

if __name__ == "__main__":
    main()
