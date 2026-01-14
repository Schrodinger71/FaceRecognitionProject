import cv2
import numpy as np
import pickle
import face_recognition
from typing import Dict, Tuple, Optional
import warnings
warnings.filterwarnings("ignore")

class FaceRecognizer:
    def __init__(self, use_svm: bool = True):
        from config import Config
        self.config = Config
        self.use_svm = use_svm
        
        # Загружаем модели
        self.load_models()
        
    def load_models(self):
        """Загрузка обученных моделей"""
        # Загружаем классификатор SVM
        if self.use_svm:
            with open(self.config.CLASSIFIER_FILE, 'rb') as f:
                self.classifier = pickle.load(f)
        
        # Загружаем центроиды
        centroids_file = os.path.join(self.config.MODELS_DIR, 'centroids.pkl')
        with open(centroids_file, 'rb') as f:
            self.centroids = pickle.load(f)
        
        print("Модели загружены успешно")
    
    def recognize_faces(self, frame: np.ndarray) -> Tuple[np.ndarray, list]:
        """Распознавание лиц на кадре"""
        # Изменение размера для скорости
        small_frame = cv2.resize(
            frame,
            (0, 0),
            fx=self.config.SCALE_FACTOR,
            fy=self.config.SCALE_FACTOR
        )
        
        # Конвертация BGR -> RGB
        rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        
        # Детекция лиц
        face_locations = face_recognition.face_locations(rgb_small)
        
        # Извлечение эмбеддингов
        face_encodings = face_recognition.face_encodings(
            rgb_small,
            known_face_locations=face_locations
        )
        
        results = []
        
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            # Масштабируем координаты обратно
            top = int(top / self.config.SCALE_FACTOR)
            right = int(right / self.config.SCALE_FACTOR)
            bottom = int(bottom / self.config.SCALE_FACTOR)
            left = int(left / self.config.SCALE_FACTOR)
            
            # Распознавание
            if self.use_svm:
                # Используем SVM
                probs = self.classifier.predict_proba([face_encoding])[0]
                confidence = np.max(probs)
                label_idx = np.argmax(probs)
                
                # Если уверенность низкая - помечаем как неизвестного
                if confidence < 0.6:
                    name = "Неизвестный"
                    confidence = 1 - confidence
                else:
                    name = self.config.LABELS.get(label_idx, f"Class_{label_idx}")
            else:
                # Используем расстояния до центроидов
                distances = {}
                for label, centroid in self.centroids.items():
                    distances[label] = np.linalg.norm(face_encoding - centroid)
                
                best_label = min(distances, key=distances.get)
                best_distance = distances[best_label]
                
                if best_distance > self.config.DISTANCE_THRESHOLD:
                    name = "Неизвестный"
                    confidence = 1 - (best_distance / 2.0)  # Нормализация
                else:
                    name = self.config.LABELS.get(best_label, f"Class_{best_label}")
                    confidence = 1 - (best_distance / self.config.DISTANCE_THRESHOLD)
            
            results.append({
                'location': (top, right, bottom, left),
                'name': name,
                'confidence': confidence
            })
        
        return frame, results
    
    def draw_results(self, frame: np.ndarray, results: list) -> np.ndarray:
        """Отрисовка результатов на кадре"""
        for result in results:
            top, right, bottom, left = result['location']
            name = result['name']
            confidence = result['confidence']
            
            # Выбор цвета
            if name == "Александр":
                color = (0, 255, 0)  # Зеленый
            elif name == "Егор":
                color = (255, 0, 0)  # Синий
            else:
                color = (0, 255, 255)  # Желтый
            
            # Рисуем рамку
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            
            # Рисуем подложку для текста
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
            
            # Добавляем текст
            cv2.putText(frame, f"{name} ({confidence:.1%})", 
                       (left + 6, bottom - 6),
                       cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)
        
        return frame
