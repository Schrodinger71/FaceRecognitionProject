import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import customtkinter as ctk
import cv2
from PIL import Image, ImageTk
import threading
import time
from datetime import datetime
import os

# Настройка темы
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

class FaceRecognitionApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        
        from config import Config
        from src.face_recognizer import FaceRecognizer
        from src.dataset_utils import DatasetManager
        from src.train_model import FaceTrainer
        
        self.config = Config
        self.recognizer = None
        self.dataset_manager = DatasetManager()
        self.trainer = FaceTrainer()
        
        self.setup_ui()
        self.is_running = False
        self.cap = None
        
    def setup_ui(self):
        """Настройка пользовательского интерфейса"""
        self.title("Система распознавания лиц - Александр и Егор")
        self.geometry(self.config.WINDOW_SIZE)
        
        # Создаем контейнеры
        self.main_container = ctk.CTkFrame(self)
        self.main_container.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Левая панель (управление)
        self.left_panel = ctk.CTkFrame(self.main_container, width=300)
        self.left_panel.pack(side="left", fill="y", padx=(0, 10))
        
        # Правая панель (видео и информация)
        self.right_panel = ctk.CTkFrame(self.main_container)
        self.right_panel.pack(side="right", fill="both", expand=True)
        
        # ===== ЛЕВАЯ ПАНЕЛЬ =====
        ctk.CTkLabel(self.left_panel, text="Управление системой", 
                     font=("Arial", 20, "bold")).pack(pady=20)
        
        # Статус
        self.status_label = ctk.CTkLabel(self.left_panel, text="Статус: Остановлен")
        self.status_label.pack(pady=10)
        
        # Кнопки управления
        self.start_btn = ctk.CTkButton(self.left_panel, text="Запуск камеры", 
                                       command=self.toggle_camera,
                                       height=40)
        self.start_btn.pack(pady=10, padx=20, fill="x")
        
        # Статистика датасета
        self.stats_frame = ctk.CTkFrame(self.left_panel)
        self.stats_frame.pack(pady=20, padx=10, fill="x")
        
        ctk.CTkLabel(self.stats_frame, text="Статистика датасета:", 
                     font=("Arial", 14, "bold")).pack(pady=5)
        
        self.stats_text = ctk.CTkTextbox(self.stats_frame, height=150)
        self.stats_text.pack(pady=10, padx=10, fill="both")
        self.update_stats()
        
        # Кнопки управления датасетом
        ctk.CTkButton(self.left_panel, text="Захватить фото Александра",
                      command=lambda: self.capture_for_person("Александр")).pack(pady=5, padx=20, fill="x")
        
        ctk.CTkButton(self.left_panel, text="Захватить фото Егора",
                      command=lambda: self.capture_for_person("Егор")).pack(pady=5, padx=20, fill="x")
        
        ctk.CTkButton(self.left_panel, text="Обновить модель",
                      command=self.train_model).pack(pady=10, padx=20, fill="x")
        
        ctk.CTkButton(self.left_panel, text="Экспорт результатов",
                      command=self.export_results).pack(pady=5, padx=20, fill="x")
        
        # Настройки
        self.settings_frame = ctk.CTkFrame(self.left_panel)
        self.settings_frame.pack(pady=20, padx=10, fill="x")
        
        ctk.CTkLabel(self.settings_frame, text="Настройки:", 
                     font=("Arial", 14, "bold")).pack(pady=5)
        
        # Порог распознавания
        self.threshold_label = ctk.CTkLabel(self.settings_frame, 
                                           text=f"Порог: {self.config.DISTANCE_THRESHOLD}")
        self.threshold_label.pack()
        
        self.threshold_slider = ctk.CTkSlider(self.settings_frame, from_=0.3, to=0.8,
                                             command=self.update_threshold)
        self.threshold_slider.set(self.config.DISTANCE_THRESHOLD)
        self.threshold_slider.pack(pady=5, padx=10, fill="x")
        
        # ===== ПРАВАЯ ПАНЕЛЬ =====
        # Видео окно
        self.video_frame = ctk.CTkFrame(self.right_panel)
        self.video_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        self.video_label = ctk.CTkLabel(self.video_frame, text="Камера не запущена")
        self.video_label.pack(fill="both", expand=True)
        
        # Панель логов
        self.log_frame = ctk.CTkFrame(self.right_panel, height=150)
        self.log_frame.pack(fill="x", padx=10, pady=(0, 10))
        
        ctk.CTkLabel(self.log_frame, text="Журнал событий:", 
                     font=("Arial", 14, "bold")).pack(anchor="w", padx=10, pady=5)
        
        self.log_text = ctk.CTkTextbox(self.log_frame, height=100)
        self.log_text.pack(padx=10, pady=(0, 10), fill="both")
        self.log_text.configure(state="disabled")
        
    def log_message(self, message: str):
        """Добавление сообщения в лог"""
        self.log_text.configure(state="normal")
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.insert("end", f"[{timestamp}] {message}\n")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")
    
    def update_stats(self):
        """Обновление статистики датасета"""
        stats = self.dataset_manager.get_dataset_stats()
        self.stats_text.configure(state="normal")
        self.stats_text.delete("1.0", "end")
        
        for person, count in stats.items():
            self.stats_text.insert("end", f"• {person}: {count} фото\n")
        
        total = sum(stats.values())
        self.stats_text.insert("end", f"\nВсего: {total} фото")
        self.stats_text.configure(state="disabled")
    
    def capture_for_person(self, person_name: str):
        """Захват фото для указанного человека"""
        def capture_thread():
            self.dataset_manager.capture_photos(person_name, num_photos=30)
            self.update_stats()
            self.log_message(f"Завершен захват фото для {person_name}")
        
        thread = threading.Thread(target=capture_thread, daemon=True)
        thread.start()
        self.log_message(f"Начат захват фото для {person_name}...")
    
    def train_model(self):
        """Обучение модели"""
        def train_thread():
            try:
                self.log_message("Начато обучение модели...")
                self.trainer.extract_embeddings()
                self.trainer.train_classifier()
                self.trainer.compute_centroids()
                self.recognizer = None  # Сброс для перезагрузки
                self.log_message("Обучение завершено успешно!")
                messagebox.showinfo("Успех", "Модель успешно обучена!")
            except Exception as e:
                self.log_message(f"Ошибка обучения: {str(e)}")
                messagebox.showerror("Ошибка", f"Ошибка обучения: {str(e)}")
        
        thread = threading.Thread(target=train_thread, daemon=True)
        thread.start()
    
    def update_threshold(self, value: float):
        """Обновление порога распознавания"""
        self.config.DISTANCE_THRESHOLD = round(value, 2)
        self.threshold_label.configure(text=f"Порог: {self.config.DISTANCE_THRESHOLD}")
        if self.recognizer:
            self.recognizer.config.DISTANCE_THRESHOLD = self.config.DISTANCE_THRESHOLD
    
    def toggle_camera(self):
        """Включение/выключение камеры"""
        if not self.is_running:
            self.start_camera()
        else:
            self.stop_camera()
    
    def start_camera(self):
        """Запуск камеры"""
        try:
            if self.recognizer is None:
                from src.face_recognizer import FaceRecognizer
                self.recognizer = FaceRecognizer()
            
            self.cap = cv2.VideoCapture(self.config.CAMERA_INDEX)
            if not self.cap.isOpened():
                raise RuntimeError("Не удалось открыть камеру")
            
            self.is_running = True
            self.start_btn.configure(text="Остановить камеру")
            self.status_label.configure(text="Статус: Запущена", text_color="green")
            self.log_message("Камера запущена")
            
            # Запуск потока обработки видео
            self.video_thread = threading.Thread(target=self.process_video, daemon=True)
            self.video_thread.start()
            
        except Exception as e:
            self.log_message(f"Ошибка запуска камеры: {str(e)}")
            messagebox.showerror("Ошибка", f"Не удалось запустить камеру: {str(e)}")
    
    def stop_camera(self):
        """Остановка камеры"""
        self.is_running = False
        self.start_btn.configure(text="Запуск камеры")
        self.status_label.configure(text="Статус: Остановлен", text_color="red")
        
        if self.cap:
            self.cap.release()
            self.cap = None
        
        self.video_label.configure(text="Камера не запущена")
        self.log_message("Камера остановлена")
    
    def process_video(self):
        """Обработка видео потока"""
        recognition_count = {"Александр": 0, "Егор": 0, "Неизвестный": 0}
        
        while self.is_running and self.cap:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # Распознавание лиц
            processed_frame, results = self.recognizer.recognize_faces(frame)
            
            # Обновление статистики
            for result in results:
                name = result['name']
                if name in recognition_count:
                    recognition_count[name] += 1
            
            # Отрисовка результатов
            processed_frame = self.recognizer.draw_results(processed_frame, results)
            
            # Конвертация для tkinter
            rgb_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)
            
            # Изменение размера под окно
            window_width = self.video_label.winfo_width()
            window_height = self.video_label.winfo_height()
            
            if window_width > 1 and window_height > 1:
                pil_image = pil_image.resize((window_width, window_height), Image.LANCZOS)
            
            tk_image = ImageTk.PhotoImage(pil_image)
            
            # Обновление изображения
            self.video_label.configure(image=tk_image)
            self.video_label.image = tk_image
            
            # Добавляем небольшую задержку
            time.sleep(0.03)
    
    def export_results(self):
        """Экспорт результатов распознавания"""
        filename = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write("Отчет системы распознавания лиц\n")
                    f.write("=" * 40 + "\n")
                    f.write(f"Дата создания: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write("\nСтатистика датасета:\n")
                    
                    stats = self.dataset_manager.get_dataset_stats()
                    for person, count in stats.items():
                        f.write(f"  {person}: {count} фото\n")
                    
                    f.write("\nНастройки системы:\n")
                    f.write(f"  Порог распознавания: {self.config.DISTANCE_THRESHOLD}\n")
                    f.write(f"  Масштаб обработки: {self.config.SCALE_FACTOR}\n")
                
                self.log_message(f"Отчет экспортирован в {filename}")
                messagebox.showinfo("Успех", f"Отчет успешно сохранен в:\n{filename}")
            except Exception as e:
                self.log_message(f"Ошибка экспорта: {str(e)}")
                messagebox.showerror("Ошибка", f"Ошибка экспорта: {str(e)}")
    
    def on_closing(self):
        """Обработка закрытия окна"""
        self.stop_camera()
        self.destroy()
