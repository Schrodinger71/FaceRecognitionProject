import os
import cv2
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog, messagebox
from datetime import datetime
from typing import List, Tuple, Optional
import warnings
warnings.filterwarnings("ignore")

class FileProcessor:
    def __init__(self, recognizer):
        """
        Инициализация процессора файлов
        
        Args:
            recognizer: Объект FaceRecognizer для распознавания лиц
        """
        from config import Config
        self.config = Config
        self.recognizer = recognizer
    
    def process_single_image(self, image_path: str, save_result: bool = True):
        """
        Обработка одного изображения
        
        Args:
            image_path: Путь к изображению
            save_result: Сохранять ли результат
        
        Returns:
            tuple: (обработанное изображение, результаты распознавания)
        """
        # Чтение изображения
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Не удалось загрузить изображение: {image_path}")
        
        # Распознавание лиц
        processed_image, results = self.recognizer.recognize_faces(image)
        
        # Отрисовка результатов
        processed_image = self.recognizer.draw_results(processed_image, results)
        
        # Сохранение результата
        if save_result and results:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.basename(image_path)
            result_path = os.path.join(
                self.config.RESULTS_DIR, 
                "images", 
                f"result_{timestamp}_{filename}"
            )
            cv2.imwrite(result_path, processed_image)
            print(f"Результат сохранен: {result_path}")
        
        return processed_image, results
    
    def process_video_file(self, video_path: str, output_path: Optional[str] = None):
        """
        Обработка видеофайла
        
        Args:
            video_path: Путь к видеофайлу
            output_path: Путь для сохранения результата
        
        Returns:
            str: Путь к обработанному видео
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Не удалось открыть видео: {video_path}")
        
        # Получаем параметры видео
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Создаем путь для сохранения
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.basename(video_path)
            output_path = os.path.join(
                self.config.RESULTS_DIR,
                "videos",
                f"processed_{timestamp}_{filename}"
            )
        
        # Создаем VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        print(f"Обработка видео: {video_path}")
        print(f"Размер: {width}x{height}, FPS: {fps}")
        
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Распознавание лиц в кадре
            processed_frame, results = self.recognizer.recognize_faces(frame)
            processed_frame = self.recognizer.draw_results(processed_frame, results)
            
            # Запись кадра
            out.write(processed_frame)
            
            frame_count += 1
            if frame_count % 30 == 0:
                print(f"Обработано кадров: {frame_count}")
            
            # Показываем предпросмотр (каждый 10-й кадр)
            if frame_count % 10 == 0:
                preview = cv2.resize(processed_frame, (640, 480))
                cv2.imshow('Обработка видео', preview)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        
        print(f"Видео обработано. Сохранено в: {output_path}")
        return output_path
    
    def process_directory(self, directory_path: str):
        """
        Обработка всех изображений в директории
        
        Args:
            directory_path: Путь к директории
        
        Returns:
            dict: Статистика распознавания
        """
        statistics = {
            "total_images": 0,
            "processed": 0,
            "faces_found": 0,
            "recognitions": {}
        }
        
        # Инициализация счетчиков для каждого человека
        for label in self.config.LABELS.values():
            statistics["recognitions"][label] = 0
        
        # Поиск изображений
        image_files = []
        for file in os.listdir(directory_path):
            if file.lower().endswith(self.config.IMAGE_EXTENSIONS):
                image_files.append(os.path.join(directory_path, file))
        
        statistics["total_images"] = len(image_files)
        
        if not image_files:
            print("Изображения не найдены")
            return statistics
        
        print(f"Найдено {len(image_files)} изображений")
        
        # Обработка каждого изображения
        for i, image_path in enumerate(image_files, 1):
            try:
                print(f"Обработка {i}/{len(image_files)}: {os.path.basename(image_path)}")
                
                _, results = self.process_single_image(image_path, save_result=True)
                
                statistics["processed"] += 1
                statistics["faces_found"] += len(results)
                
                # Подсчет распознаваний
                for result in results:
                    name = result['name']
                    if name in statistics["recognitions"]:
                        statistics["recognitions"][name] += 1
                    else:
                        statistics["recognitions"][name] = 1
                
            except Exception as e:
                print(f"Ошибка обработки {image_path}: {e}")
        
        return statistics
    
    def extract_faces_from_image(self, image_path: str, output_dir: str):
        """
        Извлечение и сохранение отдельных лиц из изображения
        
        Args:
            image_path: Путь к изображению
            output_dir: Директория для сохранения лиц
        
        Returns:
            list: Пути к сохраненным лицам
        """
        os.makedirs(output_dir, exist_ok=True)
        
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Не удалось загрузить изображение: {image_path}")
        
        # Изменение размера для скорости
        small_frame = cv2.resize(
            image,
            (0, 0),
            fx=self.config.SCALE_FACTOR,
            fy=self.config.SCALE_FACTOR
        )
        
        # Детекция лиц
        rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_small)
        
        saved_faces = []
        
        for i, (top, right, bottom, left) in enumerate(face_locations):
            # Масштабируем координаты обратно
            top = int(top / self.config.SCALE_FACTOR)
            right = int(right / self.config.SCALE_FACTOR)
            bottom = int(bottom / self.config.SCALE_FACTOR)
            left = int(left / self.config.SCALE_FACTOR)
            
            # Вырезаем лицо
            face = image[top:bottom, left:right]
            
            if face.size == 0:
                continue
            
            # Сохраняем лицо
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            face_filename = f"face_{timestamp}_{i}.jpg"
            face_path = os.path.join(output_dir, face_filename)
            
            cv2.imwrite(face_path, face)
            saved_faces.append(face_path)
            
            print(f"Сохранено лицо {i+1}: {face_path}")
        
        return saved_faces
    
    def create_report(self, statistics: dict, output_file: str = None):
        """
        Создание отчета о распознавании
        
        Args:
            statistics: Статистика распознавания
            output_file: Путь для сохранения отчета
        
        Returns:
            str: Текст отчета
        """
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = os.path.join(self.config.RESULTS_DIR, f"report_{timestamp}.txt")
        
        report_lines = [
            "=" * 50,
            "ОТЧЕТ О РАСПОЗНАВАНИИ ЛИЦ",
            f"Дата: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "=" * 50,
            "",
            f"Всего изображений: {statistics['total_images']}",
            f"Обработано: {statistics['processed']}",
            f"Найдено лиц: {statistics['faces_found']}",
            "",
            "РАСПРЕДЕЛЕНИЕ РАСПОЗНАВАНИЙ:",
            "-" * 30,
        ]
        
        # Добавляем статистику по каждому человеку
        for name, count in statistics["recognitions"].items():
            if count > 0:
                report_lines.append(f"{name:20}: {count:4} раз")
        
        report_text = "\n".join(report_lines)
        
        # Сохраняем в файл
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        print(f"Отчет сохранен: {output_file}")
        return report_text
    
    def convert_to_tk_image(self, cv2_image: np.ndarray, max_size: Tuple[int, int] = (800, 600)):
        """
        Конвертация изображения OpenCV для отображения в tkinter
        
        Args:
            cv2_image: Изображение OpenCV (BGR)
            max_size: Максимальный размер (ширина, высота)
        
        Returns:
            ImageTk.PhotoImage: Изображение для tkinter
        """
        # Конвертация BGR -> RGB
        rgb_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
        
        # Конвертация в PIL Image
        pil_image = Image.fromarray(rgb_image)
        
        # Изменение размера
        pil_image.thumbnail(max_size, Image.LANCZOS)
        
        # Конвертация в PhotoImage
        return ImageTk.PhotoImage(pil_image)
    
    def get_file_size(self, file_path: str):
        """
        Получение размера файла в удобном формате
        
        Args:
            file_path: Путь к файлу
        
        Returns:
            str: Размер файла (KB, MB, GB)
        """
        size_bytes = os.path.getsize(file_path)
        
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024.0
        
        return f"{size_bytes:.1f} TB"
