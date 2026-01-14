import os
import cv2
import numpy as np
from PIL import Image
import shutil
from tqdm import tqdm
import warnings
from typing import Dict, List, Optional, Tuple, Any
warnings.filterwarnings("ignore")

class DatasetManager:
    def __init__(self):
        from config import Config
        self.config = Config
    
    def capture_photos(self, person_name: str, num_photos: int = 30) -> int:
        """
        Захват фото с веб-камеры для указанного человека
        
        Args:
            person_name: Имя человека (Александр или Егор)
            num_photos: Количество фото для захвата
        
        Returns:
            int: Количество сохраненных фото
        """
        person_dir = os.path.join(self.config.DATASET_DIR, person_name)
        os.makedirs(person_dir, exist_ok=True)
        
        cap = cv2.VideoCapture(self.config.CAMERA_INDEX, cv2.CAP_DSHOW)
        if not cap.isOpened():
            raise RuntimeError("Не удалось открыть камеру")
        
        print(f"📸 Захват {num_photos} фото для {person_name}")
        print("Нажмите 'q' для выхода или 'c' для ручного захвата")
        
        count = 0
        while count < num_photos:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Показываем кадр с инструкциями
            display = frame.copy()
            cv2.putText(display, f"{person_name}: {count+1}/{num_photos}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(display, "Нажмите 'c' для снимка, 'q' для выхода", 
                       (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow(f'Захват фото - {person_name}', display)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                # Ручной захват
                filename = os.path.join(person_dir, f"manual_{count+1:03d}.jpg")
                cv2.imwrite(filename, frame)
                count += 1
                print(f"  📸 Снимок {count}/{num_photos} сохранен")
            
            # Автоматический захват каждые 2 секунды
            if cv2.getWindowProperty(f'Захват фото - {person_name}', cv2.WND_PROP_VISIBLE) >= 1:
                if count < num_photos and count % 5 == 0:
                    filename = os.path.join(person_dir, f"auto_{count+1:03d}.jpg")
                    cv2.imwrite(filename, frame)
                    count += 1
                    print(f"  🤖 Авто-снимок {count}/{num_photos}")
        
        cap.release()
        cv2.destroyAllWindows()
        
        # Подсчитываем итоговое количество
        photos = [f for f in os.listdir(person_dir) 
                 if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        
        print(f"\n✅ Завершено! Сохранено {len(photos)} фото в {person_dir}")
        return len(photos)
    
    def get_dataset_stats(self) -> Dict[str, int]:
        """
        Получение статистики датасета
        
        Returns:
            dict: Словарь {имя_человека: количество_фото}
        """
        stats: Dict[str, int] = {}
        
        if not os.path.exists(self.config.DATASET_DIR):
            return stats
        
        for person in os.listdir(self.config.DATASET_DIR):
            person_path = os.path.join(self.config.DATASET_DIR, person)
            if os.path.isdir(person_path):
                photos = [f for f in os.listdir(person_path) 
                         if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
                stats[person] = len(photos)
        
        return stats
    
    def add_lfw_dataset(self, lfw_path: str, max_per_person: int = 10) -> int:
        """
        Добавление лиц из LFW датасета в папку 'Неизвестный'
        
        Args:
            lfw_path: Путь к папке LFW
            max_per_person: Максимальное количество фото от одного человека
        
        Returns:
            int: Количество добавленных фото
        """
        unknown_dir = os.path.join(self.config.DATASET_DIR, "Неизвестный")
        os.makedirs(unknown_dir, exist_ok=True)
        
        print(f"📂 Поиск лиц в: {lfw_path}")
        
        if not os.path.exists(lfw_path):
            print(f"❌ Путь не существует: {lfw_path}")
            return 0
        
        # Находим папки с людьми
        person_folders: List[str] = []
        for item in os.listdir(lfw_path):
            item_path = os.path.join(lfw_path, item)
            if os.path.isdir(item_path):
                # Проверяем, есть ли JPG файлы
                jpg_files = [f for f in os.listdir(item_path) 
                            if f.lower().endswith('.jpg')]
                if jpg_files:
                    person_folders.append(item)
        
        print(f"👥 Найдено людей: {len(person_folders)}")
        
        if not person_folders:
            print("⚠️  Не найдено папок с изображениями")
            return 0
        
        added = 0
        skipped = 0
        
        # Обрабатываем первых 50 человек (для скорости)
        for person_name in tqdm(person_folders[:50], desc="Добавление лиц"):
            person_path = os.path.join(lfw_path, person_name)
            
            # Получаем JPG файлы
            photos = [f for f in os.listdir(person_path) 
                     if f.lower().endswith(('.jpg', '.jpeg'))]
            photos = photos[:max_per_person]
            
            for photo in photos:
                src = os.path.join(person_path, photo)
                
                # Создаем безопасное имя файла
                safe_name = person_name.replace(" ", "_").replace("'", "")
                if len(safe_name) > 30:
                    safe_name = safe_name[:30]
                
                dst = os.path.join(unknown_dir, f"lfw_{safe_name}_{photo}")
                
                if os.path.exists(dst):
                    skipped += 1
                    continue
                
                try:
                    shutil.copy2(src, dst)
                    added += 1
                except Exception as e:
                    print(f"⚠️  Ошибка копирования {src}: {e}")
                    skipped += 1
        
        print(f"\n✅ Добавлено: {added} фото")
        print(f"⚠️  Пропущено: {skipped} фото")
        
        return added
    
    def clear_dataset(self, person_name: Optional[str] = None) -> None:
        """
        Очистка датасета
        
        Args:
            person_name: Имя человека для очистки (если None - очистить весь датасет)
        """
        if person_name:
            person_dir = os.path.join(self.config.DATASET_DIR, person_name)
            if os.path.exists(person_dir):
                for file in os.listdir(person_dir):
                    file_path = os.path.join(person_dir, file)
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                print(f"✅ Очищена папка {person_name}")
        else:
            for person in os.listdir(self.config.DATASET_DIR):
                person_dir = os.path.join(self.config.DATASET_DIR, person)
                if os.path.isdir(person_dir):
                    for file in os.listdir(person_dir):
                        file_path = os.path.join(person_dir, file)
                        if os.path.isfile(file_path):
                            os.remove(file_path)
            print("✅ Весь датасет очищен")
