import cv2
import numpy as np
import pickle
import face_recognition
import os
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings("ignore")

class FaceRecognizer:
    def __init__(self, use_svm: bool = False):
        """
        Инициализация распознавателя лиц
        
        Args:
            use_svm: Использовать SVM классификатор (False = использовать центроиды)
        """
        from config import Config
        self.config = Config
        self.use_svm = use_svm
        self.classifier: Optional[Any] = None
        self.centroids: Optional[Dict[int, np.ndarray]] = None
        self.label_names: Dict[int, str] = {}
        
        self.load_models()
    
    def load_models(self) -> None:
        """Загрузка обученных моделей"""
        try:
            # Загружаем центроиды
            if os.path.exists(self.config.CENTROIDS_FILE):
                with open(self.config.CENTROIDS_FILE, 'rb') as f:
                    self.centroids, self.label_names = pickle.load(f)
                print(f"✅ Загружены центроиды для {len(self.centroids)} классов")
            else:
                print("⚠️  Центроиды не найдены. Сначала обучите модель.")
                self.centroids = None
            
            # Загружаем SVM классификатор если нужно
            if self.use_svm and os.path.exists(self.config.CLASSIFIER_FILE):
                with open(self.config.CLASSIFIER_FILE, 'rb') as f:
                    self.classifier = pickle.load(f)
                print("✅ SVM классификатор загружен")
            elif self.use_svm:
                print("⚠️  SVM классификатор не найден. Используются центроиды.")
                self.use_svm = False
                
        except Exception as e:
            print(f"❌ Ошибка загрузки моделей: {e}")
            self.centroids = None
            self.classifier = None
    
    def recognize_faces(self, frame: np.ndarray) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        Распознавание лиц на кадре
        
        Args:
            frame: Кадр изображения (BGR)
        
        Returns:
            tuple: (оригинальный кадр, список результатов)
        """
        if self.centroids is None:
            return frame, []
        
        # Уменьшаем размер для скорости
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
        
        results: List[Dict[str, Any]] = []
        
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            # Масштабируем координаты обратно
            top = int(top / self.config.SCALE_FACTOR)
            right = int(right / self.config.SCALE_FACTOR)
            bottom = int(bottom / self.config.SCALE_FACTOR)
            left = int(left / self.config.SCALE_FACTOR)
            
            # Распознавание
            if self.use_svm and self.classifier is not None:
                # Используем SVM классификатор
                probs = self.classifier.predict_proba([face_encoding])[0]
                confidence = np.max(probs)
                label_idx = np.argmax(probs)
                
                if confidence < 0.6:  # Низкая уверенность
                    name = "Неизвестный"
                    confidence = 1 - confidence
                else:
                    name = self.config.LABELS.get(label_idx, f"Class_{label_idx}")
            
            else:
                # Используем метод центроидов (расстояний)
                distances: Dict[int, float] = {}
                for label, centroid in self.centroids.items():
                    distances[label] = np.linalg.norm(face_encoding - centroid)
                
                # Находим ближайший центроид
                best_label = min(distances, key=distances.get)
                best_distance = distances[best_label]
                
                if best_distance > self.config.DISTANCE_THRESHOLD:
                    name = "Неизвестный"
                    confidence = 1 - (best_distance / 2.0)
                else:
                    name = self.label_names.get(best_label, f"Class_{best_label}")
                    confidence = 1 - (best_distance / self.config.DISTANCE_THRESHOLD)
            
            results.append({
                'location': (top, right, bottom, left),
                'name': name,
                'confidence': confidence,
                'distance': best_distance if not self.use_svm else None
            })
        
        return frame, results
    
    def draw_results(self, frame: np.ndarray, results: List[Dict[str, Any]]) -> np.ndarray:
        """
        Отрисовка результатов на кадре
        
        Args:
            frame: Кадр для отрисовки
            results: Результаты распознавания
        
        Returns:
            np.ndarray: Кадр с отрисованными результатами
        """
        for result in results:
            top, right, bottom, left = result['location']
            name = result['name']
            confidence = result['confidence']
            
            # Выбор цвета в зависимости от имени
            if name == "Александр":
                color = (0, 255, 0)  # Зеленый
            elif name == "Егор":
                color = (255, 0, 0)  # Синий
            else:
                color = (0, 255, 255)  # Желтый
            
            # Рисуем рамку вокруг лица
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            
            # Рисуем подложку для текста
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
            
            # Добавляем текст
            text = f"{name} ({confidence:.1%})"
            cv2.putText(frame, text, 
                       (left + 6, bottom - 6),
                       cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)
        
        return frame
    
    def recognize_image_file(self, image_path: str) -> Tuple[Optional[np.ndarray], List[Dict[str, Any]]]:
        """
        Распознавание лиц на изображении из файла
        
        Args:
            image_path: Путь к изображению
        
        Returns:
            tuple: (обработанное изображение, результаты)
        """
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Не удалось загрузить изображение: {image_path}")
        
        processed_image, results = self.recognize_faces(image)
        processed_image = self.draw_results(processed_image, results)
        
        return processed_image, results
    
    def get_model_info(self) -> Dict[str, Any]:
        """Получение информации о загруженных моделях"""
        info: Dict[str, Any] = {
            "centroids_loaded": self.centroids is not None,
            "svm_loaded": self.classifier is not None,
            "num_classes": len(self.centroids) if self.centroids else 0,
            "method": "SVM" if self.use_svm and self.classifier else "Centroids"
        }
        
        if self.centroids:
            info["classes"] = list(self.label_names.values())
        
        return info
