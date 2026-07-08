"""
Спільні константи проєкту передбачення діабету.

Містить шляхи до файлів, назви ознак моделі та дозволені значення полів.
"""

from pathlib import Path

# Коренева директорія проєкту (де лежать скрипти та дані).
BASE_DIR = Path(__file__).parent

# Шлях до CSV-набору даних для навчання.
DATA_PATH = BASE_DIR / "diabetes_prediction_dataset.csv"

# Шлях до збереженого пакета всіх моделей (joblib).
MODELS_BUNDLE_PATH = BASE_DIR / "diabetes_models.joblib"

# Шлях до JSON із метриками похибки кожного алгоритму.
METRICS_PATH = BASE_DIR / "model_metrics.json"

# Зворотна сумісність: основний шлях до моделей.
MODEL_PATH = MODELS_BUNDLE_PATH

# Ознаки (features), які використовує модель.
FEATURES = [
    "gender",
    "age",
    "hypertension",
    "heart_disease",
    "smoking_history",
    "bmi",
    "HbA1c_level",
    "blood_glucose_level",
]

# Цільова змінна (label): 0 — немає діабету, 1 — є діабет.
TARGET = "diabetes"

# Допустимі значення статі (як у датасеті).
GENDERS = ("Female", "Male")

# Допустимі значення історії куріння (як у датасеті).
SMOKING_HISTORY_VALUES = (
    "never",
    "current",
    "former",
    "ever",
    "not current",
    "No Info",
)

# Українські підписи для випадаючого списку куріння у веб-формі.
SMOKING_OPTIONS_UK = {
    "never": "Ніколи",
    "current": "Зараз курю",
    "former": "Колишній курець",
    "ever": "Колись курив",
    "not current": "Зараз не курю",
    "No Info": "Немає даних",
}

# Діапазони для валідації числових полів.
VALID_RANGES = {
    "age": (1, 120),
    "bmi": (10.0, 80.0),
    "HbA1c_level": (3.0, 15.0),
    "blood_glucose_level": (50, 400),
}

# Шлях до JSON із важливістю ознак найкращої моделі.
FEATURE_IMPORTANCE_PATH = BASE_DIR / "feature_importance.json"

# Кількість топ-моделей для гіперпараметричного тюнінгу.
TUNE_TOP_N = 2

# Параметри RandomizedSearchCV.
TUNING_N_ITER = 12
TUNING_CV_FOLDS = 3
TUNING_MAX_SAMPLES = 50000

# Ваги для вибору найкращої моделі (сума = 1.0).
BEST_MODEL_WEIGHTS = {
    "roc_auc": 0.5,
    "recall": 0.3,
    "f1": 0.2,
}

# Українські назви ознак для відображення важливості.
FEATURE_LABELS_UK = {
    "gender": "Стать",
    "age": "Вік",
    "hypertension": "Гіпертонія",
    "heart_disease": "Хвороби серця",
    "smoking_history": "Куріння",
    "bmi": "ІМТ",
    "HbA1c_level": "HbA1c",
    "blood_glucose_level": "Глюкоза в крові",
}

# Поріг ймовірності для класифікації: >= значення → «діабет є».
PREDICTION_THRESHOLD = 0.5

# Діапазон слайдера порогу на формі (у частках від 0 до 1).
THRESHOLD_MIN = 0.10
THRESHOLD_MAX = 0.90
THRESHOLD_STEP_PERCENT = 5
DEFAULT_THRESHOLD_PERCENT = int(PREDICTION_THRESHOLD * 100)

# Значення форми за замовчуванням для веб-інтерфейсу.
DEFAULT_FORM = {
    "gender": "Female",
    "age": "45",
    "hypertension": "0",
    "heart_disease": "0",
    "smoking_history": "No Info",
    "bmi": "27.0",
    "HbA1c_level": "5.7",
    "blood_glucose_level": "120",
}
