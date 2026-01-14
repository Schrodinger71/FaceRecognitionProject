import os
import cv2
import numpy as np
from PIL import Image
import shutil
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

class DatasetManager:
    def __init__(self):
        from config import Config
        self.config = Config
    
    def get_dataset_stats(self):
        """Получить статистику датасета"""
        stats = {}
        for person in os.listdir(self.config.DATASET_DIR):
            person_path = os.path.join(self.config.DATASET_DIR, person)
            if os.path.isdir(person_path):
                photos = [f for f in os.listdir(person_path) if f.endswith(('.jpg', '.png', '.jpeg'))]
                stats[person] = len(photos)
        return stats

    def capture_photos(self, person_name: str, num_photos: int = 50):
        """Захват фото с веб-камеры для конкретного человека"""
        person_dir = os.path.join(self.config.DATASET_DIR, person_name)
        os.makedirs(person_dir, exist_ok=True)
        
        cap = cv2.VideoCapture(self.config.CAMERA_INDEX)
        if not cap.isOpened():
            raise RuntimeError("Не удалось открыть камеру")
        
        print(f"Захват {num_photos} фото для {person_name}. Нажмите 'q' для выхода.")
        
        count = 0
        while count < num_photos:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Показываем предпросмотр
            preview = cv2.resize(frame, (640, 480))
            cv2.putText(preview, f"Фото {count+1}/{num_photos}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(preview, f"Человек: {person_name}", 
                       (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('Захват фото', preview)
            
            # Автоматически сохраняем каждые 5 кадров
            if count % 5 == 0:
                filename = os.path.join(person_dir, f"photo_{count}.jpg")
                cv2.imwrite(filename, frame)
                count += 1
                print(f"Сохранено фото {count}/{num_photos}")
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        print(f"Завершено! Сохранено {count} фото в {person_dir}")

    def add_lfw_dataset(self, lfw_path: str, max_per_person: int = 10):
        """Добавить лица из LFW датасета в папку 'Неизвестный'"""
        unknown_dir = os.path.join(self.config.DATASET_DIR, "Неизвестный")
        os.makedirs(unknown_dir, exist_ok=True)
        
        print(f"📂 Ищем лица в: {lfw_path}")
        
        # Проверяем, существует ли путь
        if not os.path.exists(lfw_path):
            print(f"❌ Путь не существует: {lfw_path}")
            return 0
        
        # Проверяем структуру - должны быть папки с именами людей
        items = os.listdir(lfw_path)
        person_folders = []
        
        for item in items:
            item_path = os.path.join(lfw_path, item)
            if os.path.isdir(item_path):
                # Проверяем, есть ли в папке jpg файлы
                jpg_files = [f for f in os.listdir(item_path) if f.lower().endswith('.jpg')]
                if jpg_files:
                    person_folders.append(item)
        
        print(f"👥 Найдено людей с фото: {len(person_folders)}")
        
        if len(person_folders) == 0:
            print("⚠️  В папке нет подпапок с jpg файлами")
            print("Пример содержимого первых 10 элементов:")
            for item in items[:10]:
                item_path = os.path.join(lfw_path, item)
                if os.path.isdir(item_path):
                    files = os.listdir(item_path)[:3]
                    print(f"  {item}/: {', '.join(files)}...")
                else:
                    print(f"  {item} (файл)")
            return 0
        
        added = 0
        skipped = 0
        
        # Используем tqdm для прогресс-бара
        for person_name in tqdm(person_folders[:100], desc="Добавление лиц"):  # Ограничиваем 100 людьми
            person_path = os.path.join(lfw_path, person_name)
            
            # Получаем все JPG файлы
            photos = [f for f in os.listdir(person_path) 
                     if f.lower().endswith(('.jpg', '.jpeg'))]
            
            # Ограничиваем количество фото от одного человека
            photos = photos[:max_per_person]
            
            for photo in photos:
                src = os.path.join(person_path, photo)
                
                # Создаем безопасное имя файла
                safe_person_name = person_name.replace(" ", "_").replace("'", "").replace('"', "")
                # Укорачиваем слишком длинные имена
                if len(safe_person_name) > 50:
                    safe_person_name = safe_person_name[:50]
                
                dst = os.path.join(unknown_dir, f"lfw_{safe_person_name}_{photo}")
                
                # Пропускаем если файл уже существует
                if os.path.exists(dst):
                    skipped += 1
                    continue
                
                try:
                    # Копируем фото
                    shutil.copy2(src, dst)
                    added += 1
                    
                    # Проверяем, можно ли открыть изображение
                    try:
                        img = cv2.imread(dst)
                        if img is None:
                            os.remove(dst)  # Удаляем битое изображение
                            added -= 1
                            skipped += 1
                    except:
                        os.remove(dst)
                        added -= 1
                        skipped += 1
                        
                except Exception as e:
                    print(f"\n⚠️ Ошибка копирования {src}: {e}")
                    skipped += 1
        
        print("\n" + "=" * 50)
        print(f"✅ Добавлено: {added} новых фото")
        print(f"⚠️  Пропущено: {skipped} фото")
        
        # Показываем примеры
        print("\nПримеры добавленных файлов (первые 5):")
        unknown_files = os.listdir(unknown_dir)[:5]
        for file in unknown_files:
            file_path = os.path.join(unknown_dir, file)
            size_kb = os.path.getsize(file_path) / 1024
            print(f"  • {file} ({size_kb:.1f} KB)")
        
        return added
