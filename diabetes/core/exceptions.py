"""
Користувацькі винятки проєкту передбачення діабету.

Ієрархія дозволяє перехоплювати помилки на різних рівнях:
- DiabetesProjectError — усі помилки проєкту;
- InvalidPatientDataError — некоректні дані форми/API;
- ModelNotFoundError — моделі ще не навчені;
- DataLoadError — проблеми з CSV або навчанням;
- PredictionError — збій inferencing.
"""


class DiabetesProjectError(Exception):
    """Базовий виняток для помилок цього проєкту."""


class ModelNotFoundError(DiabetesProjectError, FileNotFoundError):
    """Модель ще не навчена або файл моделі відсутній на диску."""


class DataLoadError(DiabetesProjectError):
    """Помилка завантаження CSV, підготовки даних або навчання pipeline."""


class InvalidPatientDataError(DiabetesProjectError, ValueError):
    """Некоректні, неповні або поза діапазоном дані пацієнта."""


class PredictionError(DiabetesProjectError):
    """Помилка під час завантаження моделі або виконання predict/predict_proba."""
