import tkinter as tk
from tkinter import filedialog, messagebox
import customtkinter as ctk
import cv2
from PIL import Image, ImageTk
import threading
import time
from datetime import datetime
import os
import json
import queue
from typing import Dict, Any, Optional, List

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
        from src.file_processor import FileProcessor
        
        self.config = Config
        self.recognizer: Optional[FaceRecognizer] = None
        self.file_processor: Optional[FileProcessor] = None
        self.dataset_manager = DatasetManager()
        self.trainer = FaceTrainer()
        
        self.setup_ui()
        self.is_running = False
        self.is_monitoring = False
        self.cap: Optional[cv2.VideoCapture] = None
        self.processed_files = queue.Queue()
        
        # Переменные для оптимизации производительности
        self.cached_results: List[Dict[str, Any]] = []
        self.cached_frame_count = 0
        self.frame_counter = 0
        
        # Запускаем мониторинг папки uploads
        self.start_upload_monitor()
        
        # Загружаем модель если она существует
        self.load_recognizer()
    
    def setup_ui(self):
        """Настройка пользовательского интерфейса"""
        self.title("Система распознавания лиц - Aleksander и Egor")
        self.geometry(self.config.WINDOW_SIZE)
        
        # Создаем вкладки
        self.tabview = ctk.CTkTabview(self)
        self.tabview.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Добавляем вкладки
        self.tabview.add("Камера")
        self.tabview.add("Изображения")
        self.tabview.add("Загрузки")
        self.tabview.add("Настройки")
        
        # Настраиваем каждую вкладку
        self.setup_camera_tab()
        self.setup_images_tab()
        self.setup_uploads_tab()
        self.setup_settings_tab()
    
    def setup_camera_tab(self):
        """Настройка вкладки камеры"""
        tab = self.tabview.tab("Камера")
        
        # Панель управления
        control_frame = ctk.CTkFrame(tab)
        control_frame.pack(fill="x", padx=10, pady=10)
        
        # Кнопка запуска/остановки камеры
        self.start_btn = ctk.CTkButton(control_frame, text="🚀 Запуск камеры", 
                                       command=self.toggle_camera,
                                       height=40, width=200)
        self.start_btn.pack(side="left", padx=5)
        
        # Статус
        self.status_label = ctk.CTkLabel(control_frame, text="Статус: Остановлен")
        self.status_label.pack(side="left", padx=20)
        
        # FPS счетчик
        self.fps_label = ctk.CTkLabel(control_frame, text="FPS: 0.0", text_color="yellow")
        self.fps_label.pack(side="left", padx=20)
        
        # Видео окно
        video_frame = ctk.CTkFrame(tab)
        video_frame.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        
        self.video_label = ctk.CTkLabel(video_frame, text="")
        self.video_label.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Счетчики распознавания
        stats_frame = ctk.CTkFrame(tab)
        stats_frame.pack(fill="x", padx=10, pady=(0, 10))
        
        self.stats_labels = {}
        for name in ["Aleksander", "Egor", "Unknown"]:
            label = ctk.CTkLabel(stats_frame, text=f"{name}: 0")
            label.pack(side="left", padx=20)
            self.stats_labels[name] = label
    
    def setup_images_tab(self):
        """Настройка вкладки работы с изображениями"""
        tab = self.tabview.tab("Изображения")
        
        # Панель управления
        control_frame = ctk.CTkFrame(tab)
        control_frame.pack(fill="x", padx=10, pady=10)
        
        # Кнопки
        ctk.CTkButton(control_frame, text="📁 Выбрать изображение",
                     command=self.select_image).pack(side="left", padx=5)
        
        ctk.CTkButton(control_frame, text="📂 Выбрать папку",
                     command=self.select_folder).pack(side="left", padx=5)
        
        ctk.CTkButton(control_frame, text="🔍 Распознать",
                     command=self.process_selected).pack(side="left", padx=5)
        
        # Информация о выбранном файле
        self.file_info_label = ctk.CTkLabel(control_frame, text="Файл не выбран")
        self.file_info_label.pack(side="right", padx=10)
        
        # Панель предпросмотра
        preview_frame = ctk.CTkFrame(tab)
        preview_frame.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        
        # Оригинал
        orig_frame = ctk.CTkFrame(preview_frame)
        orig_frame.pack(side="left", fill="both", expand=True, padx=(0, 5))
        
        ctk.CTkLabel(orig_frame, text="Оригинал", 
                    font=("Arial", 14, "bold")).pack(pady=5)
        
        self.orig_image_label = ctk.CTkLabel(orig_frame, text="Выберите изображение")
        self.orig_image_label.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Результат
        result_frame = ctk.CTkFrame(preview_frame)
        result_frame.pack(side="right", fill="both", expand=True, padx=(5, 0))
        
        ctk.CTkLabel(result_frame, text="Результат", 
                    font=("Arial", 14, "bold")).pack(pady=5)
        
        self.result_image_label = ctk.CTkLabel(result_frame, text="Результат появится здесь")
        self.result_image_label.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Лог обработки
        log_frame = ctk.CTkFrame(tab, height=100)
        log_frame.pack(fill="x", padx=10, pady=(0, 10))
        
        ctk.CTkLabel(log_frame, text="Лог обработки:", 
                    font=("Arial", 12, "bold")).pack(anchor="w", padx=10, pady=5)
        
        self.image_log_text = ctk.CTkTextbox(log_frame, height=80)
        self.image_log_text.pack(padx=10, pady=(0, 10), fill="both")
        self.image_log_text.configure(state="disabled")
    
    def setup_uploads_tab(self):
        """Настройка вкладки загрузок"""
        tab = self.tabview.tab("Загрузки")
        
        # Панель управления
        control_frame = ctk.CTkFrame(tab)
        control_frame.pack(fill="x", padx=10, pady=10)
        
        # Включение/выключение автообработки
        self.auto_process_var = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(control_frame, text="Автоматическая обработка",
                       variable=self.auto_process_var).pack(side="left", padx=5)
        
        # Кнопка ручной проверки
        ctk.CTkButton(control_frame, text="🔄 Проверить сейчас",
                     command=self.check_uploads).pack(side="left", padx=5)
        
        # Кнопка очистки
        ctk.CTkButton(control_frame, text="🗑️ Очистить папку",
                     command=self.clear_uploads).pack(side="left", padx=5)
        
        # Информация о папке
        info_frame = ctk.CTkFrame(tab)
        info_frame.pack(fill="x", padx=10, pady=(0, 10))
        
        self.upload_info_label = ctk.CTkLabel(info_frame, 
                                            text=f"Папка: {self.config.UPLOADS_DIR}")
        self.upload_info_label.pack(pady=5)
        
        # Статус
        self.upload_status_label = ctk.CTkLabel(info_frame, 
                                               text="Статус: Мониторинг активен",
                                               text_color="green")
        self.upload_status_label.pack(pady=5)
        
        # Лог загрузок
        log_frame = ctk.CTkFrame(tab)
        log_frame.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        
        ctk.CTkLabel(log_frame, text="Лог загрузок:", 
                    font=("Arial", 12, "bold")).pack(anchor="w", padx=10, pady=5)
        
        self.upload_log_text = ctk.CTkTextbox(log_frame)
        self.upload_log_text.pack(padx=10, pady=(0, 10), fill="both", expand=True)
        self.upload_log_text.configure(state="disabled")
    
    def setup_settings_tab(self):
        """Настройка вкладки настроек"""
        tab = self.tabview.tab("Настройки")
        
        # Левая панель - статистика
        left_frame = ctk.CTkFrame(tab)
        left_frame.pack(side="left", fill="both", expand=True, padx=(10, 5), pady=10)
        
        ctk.CTkLabel(left_frame, text="📊 Статистика датасета", 
                    font=("Arial", 16, "bold")).pack(pady=20)
        
        self.stats_text = ctk.CTkTextbox(left_frame, height=150)
        self.stats_text.pack(padx=10, pady=10, fill="both")
        self.update_dataset_stats()
        
        ctk.CTkButton(left_frame, text="Обновить статистику",
                     command=self.update_dataset_stats).pack(pady=10)
        
        # Правая панель - управление
        right_frame = ctk.CTkFrame(tab)
        right_frame.pack(side="right", fill="both", expand=True, padx=(5, 10), pady=10)
        
        ctk.CTkLabel(right_frame, text="⚙️ Управление", 
                    font=("Arial", 16, "bold")).pack(pady=20)
        
        # Кнопки захвата фото
        ctk.CTkButton(right_frame, text="📸 Захватить фото Aleksanderа",
                     command=lambda: self.capture_photos("Aleksander"),
                     height=40).pack(pady=5, padx=20, fill="x")
        
        ctk.CTkButton(right_frame, text="📸 Захватить фото Egorа",
                     command=lambda: self.capture_photos("Egor"),
                     height=40).pack(pady=5, padx=20, fill="x")
        
        ctk.CTkButton(right_frame, text="🎓 Обновить модель",
                     command=self.train_model,
                     height=40).pack(pady=10, padx=20, fill="x")
        
        ctk.CTkButton(right_frame, text="📤 Экспорт отчета",
                     command=self.export_report,
                     height=40).pack(pady=5, padx=20, fill="x")
        
        # Настройки
        settings_frame = ctk.CTkFrame(right_frame)
        settings_frame.pack(fill="x", padx=20, pady=20)
        
        ctk.CTkLabel(settings_frame, text="Настройки распознавания:", 
                    font=("Arial", 14, "bold")).pack(pady=5)
        
        # Порог распознавания
        self.threshold_label = ctk.CTkLabel(settings_frame, 
                                           text=f"Порог: {self.config.DISTANCE_THRESHOLD}")
        self.threshold_label.pack()
        
        self.threshold_slider = ctk.CTkSlider(settings_frame, from_=0.3, to=0.8,
                                             command=self.update_threshold)
        self.threshold_slider.set(self.config.DISTANCE_THRESHOLD)
        self.threshold_slider.pack(pady=5, padx=10, fill="x")
        
        # Информация о модели
        model_frame = ctk.CTkFrame(right_frame)
        model_frame.pack(fill="x", padx=20, pady=(0, 10))
        
        ctk.CTkLabel(model_frame, text="Информация о модели:", 
                    font=("Arial", 14, "bold")).pack(pady=5)
        
        self.model_info_label = ctk.CTkLabel(model_frame, text="Модель не загружена")
        self.model_info_label.pack()
    
    def load_recognizer(self):
        """Загрузка модели распознавания"""
        try:
            from src.face_recognizer import FaceRecognizer
            self.recognizer = FaceRecognizer()
            
            if self.recognizer.centroids is not None:
                info = self.recognizer.get_model_info()
                text = f"✅ Модель загружена ({info['method']})\n"
                text += f"Классов: {info['num_classes']}"
                self.model_info_label.configure(text=text)
                self.log_message("Модель распознавания загружена")
            else:
                self.model_info_label.configure(text="❌ Модель не обучена")
                self.log_message("Модель не найдена. Сначала обучите модель.")
                
        except Exception as e:
            self.log_message(f"❌ Ошибка загрузки модели: {e}")
    
    def log_message(self, message: str):
        """Добавление сообщения в лог"""
        def update_log():
            timestamp = datetime.now().strftime("%H:%M:%S")
            log_entry = f"[{timestamp}] {message}\n"
            
            # Добавляем в основной лог (вкладка Изображения)
            self.image_log_text.configure(state="normal")
            self.image_log_text.insert("end", log_entry)
            self.image_log_text.see("end")
            self.image_log_text.configure(state="disabled")
        
        self.after(0, update_log)
    
    def log_upload_message(self, message: str):
        """Добавление сообщения в лог загрузок"""
        def update_log():
            timestamp = datetime.now().strftime("%H:%M:%S")
            
            self.upload_log_text.configure(state="normal")
            self.upload_log_text.insert("end", f"[{timestamp}] {message}\n")
            self.upload_log_text.see("end")
            self.upload_log_text.configure(state="disabled")
        
        self.after(0, update_log)
    
    def update_dataset_stats(self):
        """Обновление статистики датасета"""
        try:
            stats = self.dataset_manager.get_dataset_stats()
            
            self.stats_text.configure(state="normal")
            self.stats_text.delete("1.0", "end")
            
            total = 0
            for person, count in stats.items():
                self.stats_text.insert("end", f"• {person}: {count} фото\n")
                total += count
            
            self.stats_text.insert("end", f"\n📈 Всего: {total} фото")
            self.stats_text.configure(state="disabled")
            
        except Exception as e:
            self.log_message(f"Ошибка обновления статистики: {e}")
    
    def capture_photos(self, person_name: str):
        """Захват фото для указанного человека"""
        def capture_thread():
            try:
                self.log_message(f"Начало захвата фото для {person_name}...")
                count = self.dataset_manager.capture_photos(person_name, num_photos=30)
                self.log_message(f"✅ Захвачено {count} фото для {person_name}")
                self.update_dataset_stats()
            except Exception as e:
                self.log_message(f"❌ Ошибка захвата фото: {e}")
        
        thread = threading.Thread(target=capture_thread, daemon=True)
        thread.start()
    
    def train_model(self):
        """Обучение модели"""
        def train_thread():
            try:
                self.log_message("🎓 Начато обучение модели...")
                
                success = self.trainer.train_full_model()
                
                if success:
                    self.log_message("✅ Модель успешно обучена!")
                    messagebox.showinfo("Успех", "Модель успешно обучена!")
                    
                    # Перезагружаем модель
                    self.load_recognizer()
                else:
                    self.log_message("❌ Ошибка обучения модели")
                    messagebox.showerror("Ошибка", "Не удалось обучить модель")
                    
            except Exception as e:
                self.log_message(f"❌ Ошибка обучения: {e}")
                messagebox.showerror("Ошибка", f"Ошибка обучения: {e}")
        
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
                self.load_recognizer()
                if self.recognizer is None:
                    raise RuntimeError("Модель не загружена")
            
            self.cap = cv2.VideoCapture(self.config.CAMERA_INDEX, cv2.CAP_DSHOW)
            if not self.cap.isOpened():
                raise RuntimeError("Не удалось открыть камеру")
            
            # Устанавливаем меньшее разрешение для скорости
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.CAMERA_WIDTH)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.CAMERA_HEIGHT)
            # Устанавливаем FPS (если поддерживается)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            
            self.is_running = True
            self.start_btn.configure(text="⏹️ Остановить камеру")
            self.status_label.configure(text="Статус: Запущена", text_color="green")
            self.log_message("Камера запущена")
            
            # Инициализируем кэш результатов
            self.cached_results = []
            self.cached_frame_count = 0
            self.frame_counter = 0
            
            # Запускаем поток обработки видео
            self.video_thread = threading.Thread(target=self.process_video, daemon=True)
            self.video_thread.start()
            
        except Exception as e:
            self.log_message(f"❌ Ошибка запуска камеры: {e}")
            messagebox.showerror("Ошибка", f"Не удалось запустить камеру: {e}")
    
    def stop_camera(self):
        """Остановка камеры"""
        self.is_running = False
        self.start_btn.configure(text="🚀 Запуск камеры")
        self.status_label.configure(text="Статус: Остановлен", text_color="red")
        
        if self.cap:
            self.cap.release()
            self.cap = None
        
        self.video_label.configure(text="")
        self.log_message("Камера остановлена")
    
    def process_video(self):
        """Обработка видео потока (оптимизированная версия)"""
        recognition_count = {"Aleksander": 0, "Egor": 0, "Unknown": 0}
        last_update_time = time.time()
        fps_start_time = time.time()
        fps_frame_count = 0
        
        while self.is_running and self.cap:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            self.frame_counter += 1
            fps_frame_count += 1
            
            # Пропускаем кадры для ускорения (обрабатываем каждый N-й кадр)
            process_frame = (self.frame_counter % self.config.PROCESS_EVERY_N_FRAMES == 0)
            
            if process_frame:
                # Распознавание лиц (с уменьшенным разрешением)
                processed_frame, results = self.recognizer.recognize_faces(frame, use_scale=True)
                
                # Обновляем кэш результатов
                self.cached_results = results
                self.cached_frame_count = 0
                
                # Обновляем счетчики
                for result in results:
                    name = result['name']
                    if name in recognition_count:
                        recognition_count[name] += 1
            else:
                # Используем кэшированные результаты
                processed_frame = frame.copy()
                if self.cached_results:
                    results = self.cached_results
                    self.cached_frame_count += 1
                    # Если кэш устарел, очищаем его
                    if self.cached_frame_count > self.config.CACHE_RESULTS_FRAMES:
                        self.cached_results = []
                else:
                    results = []
            
            # Отрисовка результатов
            if results:
                processed_frame = self.recognizer.draw_results(processed_frame, results)
            
            # Обновляем GUI только с определенной частотой для экономии ресурсов
            current_time = time.time()
            if current_time - last_update_time >= self.config.GUI_UPDATE_INTERVAL:
                # Вычисляем FPS
                fps_elapsed = current_time - fps_start_time
                if fps_elapsed >= 1.0:  # Обновляем FPS раз в секунду
                    fps = fps_frame_count / fps_elapsed
                    self.fps_label.configure(text=f"FPS: {fps:.1f}")
                    fps_frame_count = 0
                    fps_start_time = current_time
                
                # Конвертация для отображения
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
                
                # Обновляем счетчики в GUI
                for name, count in recognition_count.items():
                    if name in self.stats_labels:
                        self.stats_labels[name].configure(text=f"{name}: {count}")
                
                last_update_time = current_time
            
            # Минимальная задержка для освобождения CPU
            time.sleep(0.001)
    
    def select_image(self):
        """Выбор изображения для обработки"""
        filetypes = [
            ("Изображения", "*.jpg *.jpeg *.png *.bmp"),
            ("Все файлы", "*.*")
        ]
        
        filename = filedialog.askopenfilename(
            title="Выберите изображение",
            filetypes=filetypes
        )
        
        if filename:
            self.current_image_path = filename
            self.display_original_image(filename)
            self.file_info_label.configure(text=f"Файл: {os.path.basename(filename)}")
    
    def select_folder(self):
        """Выбор папки с изображениями"""
        folder = filedialog.askdirectory(title="Выберите папку с изображениями")
        
        if folder:
            self.current_folder = folder
            self.file_info_label.configure(text=f"Папка: {os.path.basename(folder)}")
            self.process_folder(folder)
    
    def display_original_image(self, image_path: str):
        """Отображение оригинального изображения"""
        try:
            image = Image.open(image_path)
            image.thumbnail((400, 400), Image.LANCZOS)
            
            tk_image = ImageTk.PhotoImage(image)
            self.orig_image_label.configure(image=tk_image, text="")
            self.orig_image_label.image = tk_image
            
        except Exception as e:
            self.log_message(f"❌ Ошибка загрузки изображения: {e}")
    
    def process_selected(self):
        """Обработка выбранного изображения или папки"""
        if hasattr(self, 'current_image_path'):
            self.process_single_image(self.current_image_path)
        elif hasattr(self, 'current_folder'):
            self.process_folder(self.current_folder)
        else:
            messagebox.showwarning("Внимание", "Сначала выберите изображение или папку")
    
    def process_single_image(self, image_path: str):
        """Обработка одного изображения"""
        def process_thread():
            try:
                self.log_message(f"Обработка изображения: {os.path.basename(image_path)}")
                
                if self.recognizer is None:
                    self.load_recognizer()
                    if self.recognizer is None:
                        self.log_message("❌ Модель не загружена")
                        return
                
                # Обрабатываем изображение
                from src.file_processor import FileProcessor
                processor = FileProcessor(self.recognizer)
                
                result_image, results = processor.process_single_image(
                    image_path, 
                    save_result=True
                )
                
                # Отображаем результат
                rgb_image = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(rgb_image)
                pil_image.thumbnail((400, 400), Image.LANCZOS)
                
                tk_image = ImageTk.PhotoImage(pil_image)
                
                self.after(0, lambda: self.show_result_image(tk_image, results))
                
                self.log_message(f"✅ Обработано. Найдено лиц: {len(results)}")
                
            except Exception as e:
                self.log_message(f"❌ Ошибка обработки: {e}")
        
        thread = threading.Thread(target=process_thread, daemon=True)
        thread.start()
    
    def show_result_image(self, tk_image, results):
        """Отображение результата обработки"""
        self.result_image_label.configure(image=tk_image, text="")
        self.result_image_label.image = tk_image
        
        # Показываем информацию о результатах
        if results:
            names = ", ".join([r['name'] for r in results])
            self.log_message(f"📊 Распознано: {names}")
        else:
            self.log_message("📊 Лица не найдены")
    
    def process_folder(self, folder_path: str):
        """Пакетная обработка папки с изображениями"""
        def process_thread():
            try:
                self.log_message(f"Пакетная обработка папки: {os.path.basename(folder_path)}")
                
                if self.recognizer is None:
                    self.load_recognizer()
                    if self.recognizer is None:
                        self.log_message("❌ Модель не загружена")
                        return
                
                from src.file_processor import FileProcessor
                processor = FileProcessor(self.recognizer)
                
                statistics = processor.process_directory(folder_path)
                
                # Создаем отчет
                report = processor.create_report(statistics)
                
                self.log_message(f"✅ Пакетная обработка завершена")
                self.log_message(f"📊 Обработано: {statistics['processed']}/{statistics['total']}")
                self.log_message(f"📊 Найдено лиц: {statistics['faces_found']}")
                
                # Показываем статистику
                messagebox.showinfo("Результаты", 
                                  f"Обработано: {statistics['processed']}/{statistics['total']}\n"
                                  f"Найдено лиц: {statistics['faces_found']}")
                
            except Exception as e:
                self.log_message(f"❌ Ошибка пакетной обработки: {e}")
        
        thread = threading.Thread(target=process_thread, daemon=True)
        thread.start()
    
    def start_upload_monitor(self):
        """Запуск мониторинга папки uploads"""
        self.is_monitoring = True
        self.monitor_thread = threading.Thread(
            target=self.monitor_uploads,
            daemon=True
        )
        self.monitor_thread.start()
        self.log_upload_message("Мониторинг папки uploads запущен")
    
    def monitor_uploads(self):
        """Мониторинг папки uploads"""
        processed_files = set()
        
        while self.is_monitoring:
            try:
                if os.path.exists(self.config.UPLOADS_DIR):
                    current_files = set()
                    
                    # Находим все изображения
                    for file in os.listdir(self.config.UPLOADS_DIR):
                        file_path = os.path.join(self.config.UPLOADS_DIR, file)
                        if os.path.isfile(file_path):
                            ext = os.path.splitext(file)[1].lower()
                            if ext in self.config.IMAGE_EXTENSIONS:
                                current_files.add(file_path)
                    
                    # Находим новые файлы
                    new_files = current_files - processed_files
                    
                    # Обрабатываем новые файлы если автообработка включена
                    if new_files and self.auto_process_var.get():
                        for file_path in new_files:
                            self.process_uploaded_file(file_path)
                            processed_files.add(file_path)
                    
                    # Обновляем информацию о папке
                    self.after(0, self.update_upload_info)
                
                # Ждем перед следующей проверкой
                time.sleep(self.config.AUTO_PROCESS_INTERVAL)
                
            except Exception as e:
                self.log_upload_message(f"❌ Ошибка мониторинга: {e}")
                time.sleep(10)
    
    def process_uploaded_file(self, file_path: str):
        """Обработка загруженного файла"""
        try:
            filename = os.path.basename(file_path)
            self.log_upload_message(f"🔍 Обнаружен новый файл: {filename}")
            
            if self.recognizer is None:
                self.load_recognizer()
                if self.recognizer is None:
                    self.log_upload_message("❌ Модель не загружена")
                    return
            
            from src.file_processor import FileProcessor
            processor = FileProcessor(self.recognizer)
            
            # Обрабатываем файл
            result_image, results = processor.process_single_image(file_path, save_result=True)
            
            # Перемещаем в архив
            import shutil
            archive_dir = os.path.join(self.config.UPLOADS_DIR, "processed")
            os.makedirs(archive_dir, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            archive_path = os.path.join(archive_dir, f"processed_{timestamp}_{filename}")
            shutil.move(file_path, archive_path)
            
            # Логируем результат
            if results:
                names = ", ".join([r['name'] for r in results])
                self.log_upload_message(f"✅ Обработано: {filename} -> {names}")
            else:
                self.log_upload_message(f"⚠️  Обработано: {filename} -> лица не найдены")
            
        except Exception as e:
            self.log_upload_message(f"❌ Ошибка обработки {os.path.basename(file_path)}: {e}")
    
    def update_upload_info(self):
        """Обновление информации о папке uploads"""
        try:
            upload_dir = self.config.UPLOADS_DIR
            
            if os.path.exists(upload_dir):
                total_files = 0
                image_files = 0
                
                for file in os.listdir(upload_dir):
                    file_path = os.path.join(upload_dir, file)
                    if os.path.isfile(file_path):
                        total_files += 1
                        ext = os.path.splitext(file)[1].lower()
                        if ext in self.config.IMAGE_EXTENSIONS:
                            image_files += 1
                
                info_text = f"Папка: {upload_dir}\n"
                info_text += f"Файлов: {total_files}\n"
                info_text += f"Изображений: {image_files}"
                
                self.upload_info_label.configure(text=info_text)
                
                # Обновляем статус
                if self.auto_process_var.get():
                    status = "✅ Автообработка включена"
                    color = "green"
                else:
                    status = "⏸️ Автообработка отключена"
                    color = "yellow"
                
                self.upload_status_label.configure(text=f"Статус: {status}", 
                                                  text_color=color)
                
        except Exception as e:
            self.log_upload_message(f"Ошибка обновления информации: {e}")
    
    def check_uploads(self):
        """Ручная проверка папки uploads"""
        def check_thread():
            self.log_upload_message("🔄 Ручная проверка загрузок...")
            
            if not os.path.exists(self.config.UPLOADS_DIR):
                self.log_upload_message("❌ Папка uploads не существует")
                return
            
            # Находим все изображения
            files_to_process = []
            for file in os.listdir(self.config.UPLOADS_DIR):
                file_path = os.path.join(self.config.UPLOADS_DIR, file)
                if os.path.isfile(file_path):
                    ext = os.path.splitext(file)[1].lower()
                    if ext in self.config.IMAGE_EXTENSIONS:
                        files_to_process.append(file_path)
            
            if not files_to_process:
                self.log_upload_message("⚠️  Нет файлов для обработки")
                return
            
            self.log_upload_message(f"Найдено {len(files_to_process)} файлов")
            
            # Обрабатываем файлы
            for file_path in files_to_process:
                self.process_uploaded_file(file_path)
            
            self.log_upload_message("✅ Ручная проверка завершена")
        
        thread = threading.Thread(target=check_thread, daemon=True)
        thread.start()
    
    def clear_uploads(self):
        """Очистка папки uploads"""
        if messagebox.askyesno("Подтверждение", 
                              "Очистить папку uploads? Все файлы будут удалены."):
            try:
                upload_dir = self.config.UPLOADS_DIR
                
                if os.path.exists(upload_dir):
                    for file in os.listdir(upload_dir):
                        file_path = os.path.join(upload_dir, file)
                        try:
                            if os.path.isfile(file_path):
                                os.remove(file_path)
                            elif os.path.isdir(file_path) and file != "processed":
                                import shutil
                                shutil.rmtree(file_path)
                        except Exception as e:
                            self.log_upload_message(f"Ошибка удаления {file}: {e}")
                
                self.log_upload_message("✅ Папка uploads очищена")
                self.update_upload_info()
                
            except Exception as e:
                self.log_upload_message(f"❌ Ошибка очистки: {e}")
    
    def export_report(self):
        """Экспорт отчета"""
        filename = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                # Собираем информацию для отчета
                report_lines = [
                    "=" * 50,
                    "ОТЧЕТ СИСТЕМЫ РАСПОЗНАВАНИЯ ЛИЦ",
                    f"Дата: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                    "=" * 50,
                    "",
                    "СТАТИСТИКА ДАТАСЕТА:",
                    "-" * 30,
                ]
                
                # Статистика датасета
                stats = self.dataset_manager.get_dataset_stats()
                for person, count in stats.items():
                    report_lines.append(f"{person}: {count} фото")
                
                report_lines.append("")
                report_lines.append("НАСТРОЙКИ СИСТЕМЫ:")
                report_lines.append("-" * 30)
                report_lines.append(f"Порог распознавания: {self.config.DISTANCE_THRESHOLD}")
                report_lines.append(f"Масштаб обработки: {self.config.SCALE_FACTOR}")
                
                # Информация о модели
                if self.recognizer:
                    info = self.recognizer.get_model_info()
                    report_lines.append("")
                    report_lines.append("ИНФОРМАЦИЯ О МОДЕЛИ:")
                    report_lines.append("-" * 30)
                    report_lines.append(f"Метод: {info['method']}")
                    report_lines.append(f"Классов: {info['num_classes']}")
                
                # Сохраняем отчет
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write("\n".join(report_lines))
                
                self.log_message(f"✅ Отчет экспортирован: {filename}")
                messagebox.showinfo("Успех", f"Отчет сохранен:\n{filename}")
                
            except Exception as e:
                self.log_message(f"❌ Ошибка экспорта: {e}")
                messagebox.showerror("Ошибка", f"Ошибка экспорта: {e}")
    
    def on_closing(self):
        """Обработка закрытия окна"""
        self.is_running = False
        self.is_monitoring = False
        
        if self.cap:
            self.cap.release()
        
        self.destroy()
