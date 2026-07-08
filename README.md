# Передбачення діабету (ML + Flask)

Навчальний проєкт: порівняння кількох алгоритмів машинного навчання для оцінки ймовірності діабету за показниками пацієнта.

> **Увага:** модель навчальна і **не замінює** медичну діагностику.

## Можливості

- 6 алгоритмів: Random Forest, XGBoost, градієнтний бустинг, AdaBoost, дерево рішень, логістична регресія
- SMOTE на train-вибірці, метрики на test (80/20)
- композитний **рейтинг** моделей (ROC-AUC 50% + Recall 30% + F1 20%)
- гіперпараметричний тюнінг топ-2 моделей
- веб-форма з **слайдером порогу** ймовірності
- unit-тести (`pytest`)

## Структура

```
ProjectAILearning/
├── app.py                      # Flask веб-інтерфейс
├── train_diabetes_model.py     # навчання та збереження моделей
├── predict_diabetes.py         # передбачення
├── model_registry.py           # реєстр алгоритмів / pipelines
├── validators.py               # валідація даних пацієнта
├── config.py                   # шляхи та константи
├── exceptions.py               # користувацькі винятки
├── diabetes_prediction_dataset.csv
├── model_metrics.json          # метрики після навчання
├── feature_importance.json
├── templates/index.html
├── static/                     # CSS, JS
├── tests/
└── requirements.txt
```

## Вимоги

- Python 3.10+ (перевірено на 3.14)
- залежності з `requirements.txt`

## Швидкий старт

### 1. Клонування та середовище

```bash
git clone <URL-вашого-репозиторію>.git
cd ProjectAILearning

python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Навчання моделей

Артефакти `*.joblib` **не зберігаються в git** (великі файли). Перед першим запуском веб-додатка навчіть моделі:

```bash
python train_diabetes_model.py
```

Скрипт створить:

- `diabetes_models.joblib` — пакет навчених моделей
- `model_metrics.json` — метрики порівняння
- `feature_importance.json` — важливість ознак

### 3. Запуск веб-інтерфейсу

```bash
python app.py
```

Відкрийте [http://127.0.0.1:5000](http://127.0.0.1:5000).

Для debug-режиму:

```bash
# Windows PowerShell
$env:FLASK_DEBUG="1"; python app.py

# bash
FLASK_DEBUG=1 python app.py
```

Порт можна змінити змінною `PORT` (за замовчуванням `5000`).

### 4. Тести

```bash
python -m pytest tests/ -v
```

## Приклад передбачення з CLI

```bash
python predict_diabetes.py
```

## Ліцензія даних

Датасет `diabetes_prediction_dataset.csv` — публічний навчальний набір (Kaggle / подібні джерела). Перед комерційним використанням перевірте умови ліцензії оригінального набору.
