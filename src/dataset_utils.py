import os
import cv2
import numpy as np
from PIL import Image
import warnings
warnings.filterwarnings("ignore")

class DatasetManager:
    def __init__(self):
        from config import Config
        self.config = Config
        
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
    
    def get_dataset_stats(self):
        """Получить статистику датасета"""
        stats = {}
        for person in os.listdir(self.config.DATASET_DIR):
            person_path = os.path.join(self.config.DATASET_DIR, person)
            if os.path.isdir(person_path):
                photos = [f for f in os.listdir(person_path) if f.endswith(('.jpg', '.png', '.jpeg'))]
                stats[person] = len(photos)
        return stats
    
    def add_lfw_dataset(self, lfw_path: str, max_per_person: int = 20):
        """Добавить лица из LFW датасета в папку unknown"""
        unknown_dir = os.path.join(self.config.DATASET_DIR, "Неизвестный")
        os.makedirs(unknown_dir, exist_ok=True)
        
        added = 0
        for person in os.listdir(lfw_path):
            person_path = os.path.join(lfw_path, person)
            if os.path.isdir(person_path):
                photos = [f for f in os.listdir(person_path) if f.endswith('.jpg')][:max_per_person]
                for photo in photos:
                    src = os.path.join(person_path, photo)
                    dst = os.path.join(unknown_dir, f"lfw_{person}_{photo}")
                    # Копируем фото
                    import shutil
                    shutil.copy2(src, dst)
                    added += 1
        
        print(f"Добавлено {added} фото из LFW датасета")

    def add_lfw_dataset(self, lfw_path: str, max_per_person: int = 20):
        """Добавить лица из LFW датасета в папку 'Неизвестный'"""
        from config import Config
        
        unknown_dir = os.path.join(Config.DATASET_DIR, "Неизвестный")
        os.makedirs(unknown_dir, exist_ok=True)
        
        print(f"📂 Ищем лица в: {lfw_path}")
        
        added = 0
        skipped = 0
        
        # Проверяем структуру датасета
        if not os.path.exists(lfw_path):
            print(f"❌ Путь не существует: {lfw_path}")
            return
        
        # Получаем список папок (каждая папка - человек)
        person_folders = []
        for item in os.listdir(lfw_path):
            item_path = os.path.join(lfw_path, item)
            if os.path.isdir(item_path):
                person_folders.append(item)
        
        print(f"👥 Найдено людей: {len(person_folders)}")
        
        for person_name in tqdm(person_folders, desc="Обработка людей"):
            person_path = os.path.join(lfw_path, person_name)
            
            # Получаем все JPG файлы
            photos = [f for f in os.listdir(person_path) 
                    if f.lower().endswith('.jpg')]
            
            # Берем только max_per_person фото
            photos = photos[:max_per_person]
            
            for photo in photos:
                src = os.path.join(person_path, photo)
                
                # Создаем уникальное имя файла
                safe_person_name = person_name.replace(" ", "_").replace("'", "")
                dst = os.path.join(unknown_dir, f"lfw_{safe_person_name}_{photo}")
                
                # Пропускаем если файл уже существует
                if os.path.exists(dst):
                    skipped += 1
                    continue
                
                try:
                    # Копируем фото
                    shutil.copy2(src, dst)
                    added += 1
                except Exception as e:
                    print(f"⚠️ Ошибка копирования {src}: {e}")
                    skipped += 1
        
        print("\n" + "=" * 50)
        print(f"✅ Добавлено: {added} новых фото")
        print(f"⚠️  Пропущено: {skipped} фото (уже существуют)")
        print(f"📁 Всего в 'Неизвестный': {len(os.listdir(unknown_dir))} фото")
        print("=" * 50)
        
        # Показываем примеры добавленных файлов
        print("\nПримеры добавленных файлов:")
        unknown_files = os.listdir(unknown_dir)[:5]
        for file in unknown_files:
            print(f"  • {file}")
        
        return added
