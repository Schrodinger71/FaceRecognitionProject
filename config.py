import os

class Config:
    # Пути
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATASET_DIR = os.path.join(BASE_DIR, "dataset")
    MODELS_DIR = os.path.join(BASE_DIR, "models")
    
    # Настройки модели
    EMBEDDINGS_FILE = os.path.join(MODELS_DIR, "embeddings.pkl")
    CLASSIFIER_FILE = os.path.join(MODELS_DIR, "classifier.pkl")
    
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
    WINDOW_SIZE = "1200x700"
    THEME = "dark-blue"
    
    @staticmethod
    def setup_directories():
        """Создает необходимые директории"""
        os.makedirs(Config.DATASET_DIR, exist_ok=True)
        os.makedirs(Config.MODELS_DIR, exist_ok=True)
        for label in Config.LABELS.values():
            if label != "Неизвестный":
                person_dir = os.path.join(Config.DATASET_DIR, label)
                os.makedirs(person_dir, exist_ok=True)
