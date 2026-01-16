import os

class Config:
    # Пути
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATASET_DIR = os.path.join(BASE_DIR, "dataset")
    MODELS_DIR = os.path.join(BASE_DIR, "models")
    UPLOADS_DIR = os.path.join(BASE_DIR, "uploads")
    RESULTS_DIR = os.path.join(BASE_DIR, "results")
    
    # Файлы моделей
    EMBEDDINGS_FILE = os.path.join(MODELS_DIR, "embeddings.pkl")
    CLASSIFIER_FILE = os.path.join(MODELS_DIR, "classifier.pkl")
    CENTROIDS_FILE = os.path.join(MODELS_DIR, "centroids.pkl")
    
    # Настройки распознавания
    DISTANCE_THRESHOLD = 0.6
    CAMERA_INDEX = 0
    SCALE_FACTOR = 0.25  # Используется для уменьшения разрешения при обработке
    
    # Настройки производительности
    PROCESS_EVERY_N_FRAMES = 3  # Обрабатывать каждый N-й кадр (для пропуска кадров)
    CAMERA_WIDTH = 580  # Ширина камеры (меньше = быстрее)
    CAMERA_HEIGHT = 580  # Высота камеры
    CACHE_RESULTS_FRAMES = 5  # Кэшировать результаты на N кадров
    GUI_UPDATE_INTERVAL = 0.033  # Интервал обновления GUI (30 FPS)
    
    # Настройки датасета
    LABELS = {
        0: "Aleksander",
        1: "Egor",
        -1: "Unknown"
    }
    
    # Настройки GUI
    WINDOW_SIZE = "1300x800"
    THEME = "dark-blue"
    
    # Поддерживаемые форматы
    IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")
    VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv")
    
    # Настройки автообработки
    AUTO_PROCESS_INTERVAL = 5  # секунд
    
    @staticmethod
    def setup_directories():
        """Создает необходимые директории"""
        directories = [
            Config.DATASET_DIR,
            Config.MODELS_DIR,
            Config.UPLOADS_DIR,
            Config.RESULTS_DIR,
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
        
        # Подпапки для результатов
        os.makedirs(os.path.join(Config.RESULTS_DIR, "images"), exist_ok=True)
        os.makedirs(os.path.join(Config.RESULTS_DIR, "videos"), exist_ok=True)
        os.makedirs(os.path.join(Config.RESULTS_DIR, "reports"), exist_ok=True)
        
        # Папки для людей
        for label in Config.LABELS.values():
            if label != "Unknown":
                person_dir = os.path.join(Config.DATASET_DIR, label)
                os.makedirs(person_dir, exist_ok=True)
        
        # Папка для обработанных загрузок
        os.makedirs(os.path.join(Config.UPLOADS_DIR, "processed"), exist_ok=True)
