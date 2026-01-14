import os
import pickle
import numpy as np
import face_recognition
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore")

class FaceTrainer:
    def __init__(self):
        from config import Config
        self.config = Config
        
    def extract_embeddings(self):
        """Извлечение эмбеддингов из датасета"""
        X = []
        y = []
        
        for person_name in os.listdir(self.config.DATASET_DIR):
            person_path = os.path.join(self.config.DATASET_DIR, person_name)
            if not os.path.isdir(person_path):
                continue
            
            # Маппинг имени на номер класса
            if person_name in self.config.LABELS.values():
                label = list(self.config.LABELS.keys())[
                    list(self.config.LABELS.values()).index(person_name)
                ]
            else:
                label = -1  # Неизвестный
            
            print(f"Обработка: {person_name} (класс {label})")
            
            for file in os.listdir(person_path):
                if not file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    continue
                
                img_path = os.path.join(person_path, file)
                try:
                    image = face_recognition.load_image_file(img_path)
                    encodings = face_recognition.face_encodings(image)
                    
                    if len(encodings) == 1:
                        X.append(encodings[0])
                        y.append(label)
                    elif len(encodings) > 1:
                        # Если на фото несколько лиц, берем первое
                        X.append(encodings[0])
                        y.append(label)
                except Exception as e:
                    print(f"Ошибка обработки {img_path}: {e}")
        
        X = np.array(X)
        y = np.array(y)
        
        # Сохраняем эмбеддинги
        with open(self.config.EMBEDDINGS_FILE, 'wb') as f:
            pickle.dump((X, y), f)
        
        print(f"Сохранено {len(X)} эмбеддингов")
        print(f"Распределение классов: {np.bincount(y + 1)}")  # +1 для смещения отрицательных меток
        
        return X, y
    
    def train_classifier(self):
        """Обучение SVM классификатора"""
        # Загружаем эмбеддинги
        with open(self.config.EMBEDDINGS_FILE, 'rb') as f:
            X, y = pickle.load(f)
        
        # Разделяем на train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Обучаем SVM
        clf = SVC(
            kernel='linear',
            probability=True,
            class_weight='balanced'
        )
        
        clf.fit(X_train, y_train)
        
        # Оценка модели
        y_pred = clf.predict(X_test)
        report = classification_report(y_test, y_pred, target_names=[
            self.config.LABELS.get(i, f"Class_{i}") for i in np.unique(y)
        ])
        
        print("Отчет классификации:")
        print(report)
        
        # Сохраняем модель
        with open(self.config.CLASSIFIER_FILE, 'wb') as f:
            pickle.dump(clf, f)
        
        print(f"Модель сохранена в {self.config.CLASSIFIER_FILE}")
        print(f"Точность на тесте: {clf.score(X_test, y_test):.2%}")
        
        return clf
    
    def compute_centroids(self):
        """Вычисление центроидов для каждого класса"""
        with open(self.config.EMBEDDINGS_FILE, 'rb') as f:
            X, y = pickle.load(f)
        
        centroids = {}
        for label in np.unique(y):
            centroids[label] = X[y == label].mean(axis=0)
        
        # Сохраняем центроиды
        centroids_file = os.path.join(self.config.MODELS_DIR, 'centroids.pkl')
        with open(centroids_file, 'wb') as f:
            pickle.dump(centroids, f)
        
        print(f"Сохранены центроиды для {len(centroids)} классов")
        return centroids
