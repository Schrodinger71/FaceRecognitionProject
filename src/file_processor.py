import os
import cv2
import numpy as np
import json
from datetime import datetime
import shutil
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings("ignore")

class FileProcessor:
    def __init__(self, recognizer: Any):
        """
        Инициализация процессора файлов
        
        Args:
            recognizer: Объект FaceRecognizer
        """
        from config import Config
        self.config = Config
        self.recognizer = recognizer
    
    def process_single_image(self, image_path: str, save_result: bool = True) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        Обработка одного изображения
        
        Args:
            image_path: Путь к изображению
            save_result: Сохранять ли результат
        
        Returns:
            tuple: (обработанное изображение, результаты)
        """
        # Загружаем изображение
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Не удалось загрузить изображение: {image_path}")
        
        # Распознаем лица
        processed_image, results = self.recognizer.recognize_faces(image)
        processed_image = self.recognizer.draw_results(processed_image, results)
        
        # Сохраняем результат если нужно
        if save_result and results:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.basename(image_path)
            result_path = os.path.join(
                self.config.RESULTS_DIR, 
                "images", 
                f"result_{timestamp}_{filename}"
            )
            cv2.imwrite(result_path, processed_image)
        
        return processed_image, results
    
    def process_directory(self, directory_path: str) -> Dict[str, Any]:
        """
        Пакетная обработка всех изображений в директории
        
        Args:
            directory_path: Путь к директории
        
        Returns:
            dict: Статистика обработки
        """
        statistics: Dict[str, Any] = {
            "total": 0,
            "processed": 0,
            "failed": 0,
            "faces_found": 0,
            "recognitions": {}
        }
        
        # Инициализация счетчиков
        for label in self.config.LABELS.values():
            statistics["recognitions"][label] = 0
        
        # Находим все изображения
        image_files: List[str] = []
        for file in os.listdir(directory_path):
            ext = os.path.splitext(file)[1].lower()
            if ext in self.config.IMAGE_EXTENSIONS:
                image_files.append(os.path.join(directory_path, file))
        
        statistics["total"] = len(image_files)
        
        if not image_files:
            return statistics
        
        print(f"🔍 Найдено {len(image_files)} изображений для обработки")
        
        # Обрабатываем каждое изображение
        for i, image_path in enumerate(image_files, 1):
            try:
                filename = os.path.basename(image_path)
                print(f"  Обработка {i}/{len(image_files)}: {filename}")
                
                _, results = self.process_single_image(image_path, save_result=True)
                
                statistics["processed"] += 1
                statistics["faces_found"] += len(results)
                
                # Считаем распознавания
                for result in results:
                    name = result['name']
                    if name in statistics["recognitions"]:
                        statistics["recognitions"][name] += 1
                    else:
                        statistics["recognitions"][name] = 1
                        
            except Exception as e:
                statistics["failed"] += 1
                print(f"  ❌ Ошибка обработки {image_path}: {e}")
        
        return statistics
    
    def create_report(self, statistics: Dict[str, Any], output_file: Optional[str] = None) -> str:
        """
        Создание отчета
        
        Args:
            statistics: Статистика обработки
            output_file: Путь для сохранения отчета
        
        Returns:
            str: Текст отчета
        """
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = os.path.join(self.config.RESULTS_DIR, 
                                      "reports", 
                                      f"report_{timestamp}.txt")
        
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        report_lines: List[str] = [
            "=" * 50,
            "ОТЧЕТ ОБ ОБРАБОТКЕ ИЗОБРАЖЕНИЙ",
            f"Дата: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "=" * 50,
            "",
            f"Всего изображений: {statistics['total']}",
            f"Успешно обработано: {statistics['processed']}",
            f"Не удалось обработать: {statistics['failed']}",
            f"Найдено лиц всего: {statistics['faces_found']}",
            "",
            "РАСПОЗНАВАНИЯ ПО КАТЕГОРИЯМ:",
            "-" * 30,
        ]
        
        # Добавляем статистику по категориям
        for name, count in statistics["recognitions"].items():
            if count > 0:
                report_lines.append(f"{name:20}: {count:4} раз")
        
        report_text = "\n".join(report_lines)
        
        # Сохраняем в файл
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        print(f"📄 Отчет сохранен: {output_file}")
        return report_text
    
    def monitor_uploads_folder(self) -> List[Dict[str, Any]]:
        """
        Мониторинг папки uploads на новые файлы
        
        Returns:
            list: Список обработанных файлов
        """
        processed_files: List[Dict[str, Any]] = []
        
        if not os.path.exists(self.config.UPLOADS_DIR):
            return processed_files
        
        # Находим новые изображения
        for file in os.listdir(self.config.UPLOADS_DIR):
            file_path = os.path.join(self.config.UPLOADS_DIR, file)
            
            if os.path.isfile(file_path):
                ext = os.path.splitext(file)[1].lower()
                if ext in self.config.IMAGE_EXTENSIONS:
                    try:
                        # Обрабатываем файл
                        result_image, results = self.process_single_image(
                            file_path, 
                            save_result=True
                        )
                        
                        # Перемещаем в архив
                        archive_dir = os.path.join(self.config.UPLOADS_DIR, "processed")
                        os.makedirs(archive_dir, exist_ok=True)
                        
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        archive_path = os.path.join(
                            archive_dir, 
                            f"processed_{timestamp}_{file}"
                        )
                        
                        shutil.move(file_path, archive_path)
                        processed_files.append({
                            "original": file,
                            "processed": os.path.basename(archive_path),
                            "faces_found": len(results),
                            "recognitions": [r['name'] for r in results]
                        })
                        
                    except Exception as e:
                        print(f"❌ Ошибка обработки {file}: {e}")
        
        return processed_files
