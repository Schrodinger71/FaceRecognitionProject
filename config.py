import os

class Config:
    # Пути
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATASET_DIR = os.path.join(BASE_DIR, "dataset")
    MODELS_DIR = os.path.join(BASE_DIR, "models")
    UPLOADS_DIR = os.path.join(BASE_DIR, "uploads")
    RESULTS_DIR = os.path.join(BASE_DIR, "results")
    
    # Настройки модели
    EMBEDDINGS_FILE = os.path.join(MODELS_DIR, "embeddings.pkl")
    CLASSIFIER_FILE = os.path.join(MODELS_DIR, "classifier.pkl")
    CENTROIDS_FILE = os.path.join(MODELS_DIR, "centroids.pkl")
    
    # Настройки распознавания
    DISTANCE_THRESHOLD = 0.55
    CAMERA_INDEX = 0
    SCALE_FACTOR = 0.5
    
    # Настройки датасета
    LABELS = {
        0: "Александр",
        1: "Егор",
        -1: "Неизвестный"
    }
    
    # Настройки GUI
    WINDOW_SIZE = "1400x800"
    THEME = "dark-blue"
    
    # Поддерживаемые форматы файлов
    IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".gif")
    VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv")
    
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
        
        # Создаем подпапки для результатов
        os.makedirs(os.path.join(Config.RESULTS_DIR, "images"), exist_ok=True)
        os.makedirs(os.path.join(Config.RESULTS_DIR, "videos"), exist_ok=True)
        
        # Создаем папки для людей
        for label in Config.LABELS.values():
            if label != "Неизвестный":
                person_dir = os.path.join(Config.DATASET_DIR, label)
                os.makedirs(person_dir, exist_ok=True)
