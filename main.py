#!/usr/bin/env python3
"""
Главный скрипт системы распознавания лиц
Распознавание Aleksanderа и Egorа
"""

import sys
import os

# Добавляем текущую директорию в путь
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import Config
from src.gui_app import FaceRecognitionApp

def main():
    """Главная функция"""
    print("=" * 60)
    print("СИСТЕМА РАСПОЗНАВАНИЯ ЛИЦ")
    print("Цель: Распознавание Aleksanderа и Egorа")
    print("=" * 60)
    
    # Создаем необходимые директории
    Config.setup_directories()
    
    # Проверяем наличие датасета
    print("\n📊 Статистика датасета:")
    dataset_stats = {}
    
    for person in ["Aleksander", "Egor"]:
        person_dir = os.path.join(Config.DATASET_DIR, person)
        if os.path.exists(person_dir):
            photos = [f for f in os.listdir(person_dir) 
                     if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
            dataset_stats[person] = len(photos)
            print(f"  {person}: {len(photos)} фото")
        else:
            dataset_stats[person] = 0
            print(f"  {person}: 0 фото (папка не найдена)")
    
    # Предупреждение если фото мало
    if any(count < 10 for count in dataset_stats.values()):
        print("\n⚠️  ВНИМАНИЕ: рекомендуется иметь минимум 10 фото каждого человека!")
        print("   Используйте кнопку 'Захватить фото' в приложении")
    
    # Проверяем наличие моделей
    models_exist = (
        os.path.exists(Config.EMBEDDINGS_FILE) and
        os.path.exists(Config.CENTROIDS_FILE)
    )
    
    if not models_exist:
        print("\n⚠️  Модель не обучена!")
        print("   После захвата фото нажмите 'Обновить модель'")
    
    print("\n🚀 Запуск графического интерфейса...")
    
    # Запуск GUI
    app = FaceRecognitionApp()
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()

if __name__ == "__main__":
    main()
