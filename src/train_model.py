import os
import pickle
import numpy as np
import face_recognition
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from typing import Tuple, Optional, Any, Dict
import warnings
warnings.filterwarnings("ignore")

class FaceTrainer:
    def __init__(self):
        from config import Config
        self.config = Config
    
    def extract_embeddings(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Извлечение эмбеддингов из датасета
        
        Returns:
            tuple: (эмбеддинги, метки)
        """
        X: List[np.ndarray] = []  # Эмбеддинги
        y: List[int] = []  # Метки
        
        print("📊 Извлечение эмбеддингов из датасета...")
        
        for person_name in os.listdir(self.config.DATASET_DIR):
            person_path = os.path.join(self.config.DATASET_DIR, person_name)
            if not os.path.isdir(person_path):
                continue
            
            # Определяем метку класса
            if person_name == "Aleksander":
                label = 0
            elif person_name == "Egor":
                label = 1
            elif person_name == "Unknown":
                label = -1
            else:
                label = -1  # Другие папки тоже считаем неизвестными
            
            print(f"  Обработка: {person_name} (класс {label})")
            
            # Обрабатываем все изображения в папке
            processed = 0
            for file in os.listdir(person_path):
                if not file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    continue
                
                img_path = os.path.join(person_path, file)
                try:
                    # Загружаем и кодируем изображение
                    image = face_recognition.load_image_file(img_path)
                    encodings = face_recognition.face_encodings(image)
                    
                    if encodings:
                        X.append(encodings[0])  # Берем первое лицо
                        y.append(label)
                        processed += 1
                    
                except Exception as e:
                    print(f"    ❌ Ошибка {file}: {e}")
            
            print(f"    ✅ Обработано фото: {processed}")
        
        X_array = np.array(X)
        y_array = np.array(y)
        
        print(f"\n📈 Итоговая статистика:")
        print(f"  Всего эмбеддингов: {len(X_array)}")
        
        # Подсчет по классам
        unique_labels = np.unique(y_array)
        for label in unique_labels:
            count = np.sum(y_array == label)
            name = self.config.LABELS.get(label, f"Class_{label}")
            print(f"  {name}: {count} эмбеддингов")
        
        # Сохраняем эмбеддинги
        with open(self.config.EMBEDDINGS_FILE, 'wb') as f:
            pickle.dump((X_array, y_array), f)
        
        print(f"💾 Эмбеддинги сохранены в: {self.config.EMBEDDINGS_FILE}")
        
        return X_array, y_array
    
    def train_classifier(self) -> Optional[SVC]:
        """
        Обучение SVM классификатора
        
        Returns:
            SVC: Обученный классификатор
        """
        print("🎓 Обучение SVM классификатора...")
        
        # Загружаем эмбеддинги
        if not os.path.exists(self.config.EMBEDDINGS_FILE):
            print("❌ Файл с эмбеддингами не найден")
            return None
        
        with open(self.config.EMBEDDINGS_FILE, 'rb') as f:
            X, y = pickle.load(f)
        
        if len(X) < 10:
            print("❌ Недостаточно данных для обучения")
            return None
        
        # Разделяем на обучающую и тестовую выборки
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"  Размер обучающей выборки: {len(X_train)}")
        print(f"  Размер тестовой выборки: {len(X_test)}")
        
        # Создаем и обучаем SVM
        clf = SVC(
            kernel='linear',
            probability=True,
            class_weight='balanced'
        )
        
        clf.fit(X_train, y_train)
        
        # Оцениваем модель
        y_pred = clf.predict(X_test)
        accuracy = clf.score(X_test, y_test)
        
        print(f"\n📊 Результаты классификации:")
        print(f"  Точность: {accuracy:.2%}")
        
        # Подробный отчет
        target_names = [self.config.LABELS.get(i, f"Class_{i}") 
                       for i in np.unique(y)]
        report = classification_report(y_test, y_pred, 
                                      target_names=target_names)
        print(f"\nОтчет классификации:\n{report}")
        
        # Сохраняем модель
        with open(self.config.CLASSIFIER_FILE, 'wb') as f:
            pickle.dump(clf, f)
        
        print(f"💾 Модель сохранена в: {self.config.CLASSIFIER_FILE}")
        
        return clf
    
    def compute_centroids(self) -> Tuple[Optional[Dict[int, np.ndarray]], Optional[Dict[int, str]]]:
        """
        Вычисление центроидов для каждого класса
        
        Returns:
            tuple: (центроиды, имена классов)
        """
        print("🎯 Вычисление центроидов...")
        
        # Загружаем эмбеддинги
        if not os.path.exists(self.config.EMBEDDINGS_FILE):
            print("❌ Файл с эмбеддингами не найден")
            return None, None
        
        with open(self.config.EMBEDDINGS_FILE, 'rb') as f:
            X, y = pickle.load(f)
        
        # Вычисляем центроиды для каждого класса
        centroids: Dict[int, np.ndarray] = {}
        label_names: Dict[int, str] = {}
        
        unique_labels = np.unique(y)
        for label in unique_labels:
            # Эмбеддинги данного класса
            class_embeddings = X[y == label]
            
            # Вычисляем центроид (среднее значение)
            centroid = class_embeddings.mean(axis=0)
            centroids[label] = centroid
            
            # Сохраняем имя класса
            name = self.config.LABELS.get(label, f"Class_{label}")
            label_names[label] = name
            
            print(f"  {name}: центроид вычислен ({len(class_embeddings)} эмбеддингов)")
        
        # Сохраняем центроиды
        with open(self.config.CENTROIDS_FILE, 'wb') as f:
            pickle.dump((centroids, label_names), f)
        
        print(f"💾 Центроиды сохранены в: {self.config.CENTROIDS_FILE}")
        
        return centroids, label_names
    
    def train_full_model(self) -> bool:
        """
        Полный цикл обучения модели
        
        Returns:
            bool: Успешно ли обучение
        """
        print("=" * 50)
        print("ОБУЧЕНИЕ МОДЕЛИ РАСПОЗНАВАНИЯ ЛИЦ")
        print("=" * 50)
        
        try:
            # 1. Извлечение эмбеддингов
            X, y = self.extract_embeddings()
            
            if len(X) == 0:
                print("❌ Нет данных для обучения")
                return False
            
            # 2. Вычисление центроидов
            centroids, label_names = self.compute_centroids()
            
            if centroids is None:
                print("❌ Не удалось вычислить центроиды")
                return False
            
            # 3. Обучение SVM (опционально)
            if len(np.unique(y)) >= 2:  # SVM нужны минимум 2 класса
                clf = self.train_classifier()
                if clf is None:
                    print("⚠️  SVM не обучен, но центроиды готовы")
                else:
                    print("✅ SVM успешно обучен")
            
            print("\n✅ Обучение завершено успешно!")
            return True
            
        except Exception as e:
            print(f"❌ Ошибка обучения: {e}")
            return False
