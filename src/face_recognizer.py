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
        Инициализация распознавателя лиц с OpenCV детекцией
        """
        from config import Config
        self.config = Config
        self.use_svm = use_svm
        self.classifier: Optional[Any] = None
        self.centroids: Optional[Dict[int, np.ndarray]] = None
        self.label_names: Dict[int, str] = {}
        
        # Загружаем детектор лиц OpenCV
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        if os.path.exists(cascade_path):
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
            if self.face_cascade.empty():
                print("❌ Не удалось загрузить Haar Cascade")
                self.face_cascade = None
            else:
                print("✅ Haar Cascade загружен")
        else:
            print("❌ Файл Haar Cascade не найден")
            self.face_cascade = None
        
        self.load_models()
    
    def load_models(self) -> None:
        """Загрузка обученных моделей"""
        try:
            # Загружаем центроиды
            if os.path.exists(self.config.CENTROIDS_FILE):
                with open(self.config.CENTROIDS_FILE, 'rb') as f:
                    self.centroids, self.label_names = pickle.load(f)
                print(f"✅ Загружены центроиды для {len(self.centroids)} классов")
                print(f"   Классы: {list(self.label_names.values())}")
            else:
                print("⚠️  Центроиды не найдены")
                self.centroids = None
            
            # Загружаем SVM классификатор если нужно
            if self.use_svm and os.path.exists(self.config.CLASSIFIER_FILE):
                with open(self.config.CLASSIFIER_FILE, 'rb') as f:
                    self.classifier = pickle.load(f)
                print("✅ SVM классификатор загружен")
            elif self.use_svm:
                print("⚠️  SVM классификатор не найден")
                self.use_svm = False
                
        except Exception as e:
            print(f"❌ Ошибка загрузки моделей: {e}")
            self.centroids = None
            self.classifier = None
    
    def detect_faces_opencv(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Детекция лиц с использованием OpenCV"""
        if self.face_cascade is None:
            return []
        
        # Конвертируем в grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Детекция лиц
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        # Конвертируем формат (x, y, w, h) в (top, right, bottom, left)
        face_locations = []
        for (x, y, w, h) in faces:
            top = y
            right = x + w
            bottom = y + h
            left = x
            face_locations.append((top, right, bottom, left))
        
        return face_locations
    
    def recognize_faces(self, frame: np.ndarray) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        Распознавание лиц на кадре с использованием OpenCV для детекции
        """
        if self.centroids is None:
            return frame, []
        
        # Проверка на пустой кадр
        if frame is None or frame.size == 0:
            return frame, []
        
        # Детекция лиц с OpenCV
        face_locations = self.detect_faces_opencv(frame)
        
        if not face_locations:
            return frame, []
        
        # Конвертация BGR -> RGB для face_recognition
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Убеждаемся, что массив является непрерывным (contiguous) и имеет правильный dtype
        rgb_frame = np.ascontiguousarray(rgb_frame, dtype=np.uint8)
        
        # Извлечение эмбеддингов только для найденных лиц
        face_encodings = face_recognition.face_encodings(
            rgb_frame,
            known_face_locations=face_locations
        )
        
        results: List[Dict[str, Any]] = []
        
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            # Распознавание
            if self.use_svm and self.classifier is not None:
                # Используем SVM классификатор
                probs = self.classifier.predict_proba([face_encoding])[0]
                confidence = np.max(probs)
                label_idx = np.argmax(probs)
                
                if confidence < 0.6:
                    name = "Unknown"
                    confidence = 1 - confidence
                else:
                    name = self.config.LABELS.get(label_idx, f"Class_{label_idx}")
            
            else:
                # Используем метод центроидов
                if self.centroids is None:
                    continue
                    
                distances: Dict[int, float] = {}
                for label, centroid in self.centroids.items():
                    distances[label] = np.linalg.norm(face_encoding - centroid)
                
                best_label = min(distances, key=distances.get)
                best_distance = distances[best_label]
                
                if best_distance > self.config.DISTANCE_THRESHOLD:
                    name = "Unknown"
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
        """Отрисовка результатов на кадре"""
        for result in results:
            top, right, bottom, left = result['location']
            name = result['name']
            confidence = result['confidence']
            
            if name == "Aleksander":
                color = (0, 255, 0)  # Зеленый
            elif name == "Egor":
                color = (255, 0, 0)  # Синий
            elif name == "Unknown":
                color = (0, 0, 255)  # Красный
            else:
                color = (0, 255, 255)  # Желтый
            
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
            
            text = f"{name} ({confidence:.1%})"
            cv2.putText(frame, text, 
                       (left + 6, bottom - 6),
                       cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)
        
        return frame
    
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
