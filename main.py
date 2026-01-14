#!/usr/bin/env python3
"""
Главный скрипт системы распознавания лиц
Александр (person_0) и Егор (person_1)
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
    print("Система распознавания лиц")
    print("Цель: Распознавание Александра и Егора")
    print("=" * 60)
    
    # Создаем необходимые директории
    Config.setup_directories()
    
    # Проверяем наличие датасета
    dataset_stats = {}
    for person in ["Александр", "Егор"]:
        person_dir = os.path.join(Config.DATASET_DIR, person)
        if os.path.exists(person_dir):
            photos = [f for f in os.listdir(person_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
            dataset_stats[person] = len(photos)
    
    print("\nСтатистика датасета:")
    for person, count in dataset_stats.items():
        print(f"  {person}: {count} фото")
    
    if any(count < 10 for count in dataset_stats.values()):
        print("\n⚠️  Внимание: рекомендуется иметь минимум 10 фото каждого человека!")
        print("   Используйте опцию 'Захватить фото' в приложении")
    
    # Запуск GUI
    print("\nЗапуск графического интерфейса...")
    app = FaceRecognitionApp()
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()

if __name__ == "__main__":
    main()
