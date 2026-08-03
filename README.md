# Передбачення діабету (ML + Flask)

Навчальний проєкт: порівняння кількох алгоритмів машинного навчання для оцінки ймовірності діабету за показниками пацієнта.

> **Увага:** модель навчальна і **не замінює** медичну діагностику.

## Скріншоти

#### Головна сторінка (метрики + форма)

![Flask: метрики алгоритмів і форма пацієнта](docs/screenshots/flask-01-metrics-and-form.png)

#### Результати передбачення

![Flask: donut-діаграми та порівняння моделей](docs/screenshots/flask-02-prediction-results.png)

## Можливості

- 6 алгоритмів: Random Forest, XGBoost, градієнтний бустинг, AdaBoost, дерево рішень, логістична регресія
- **3-way split** (train / validation / test): вибір і тюнінг на validation, фінальні метрики на test
- SMOTE лише для логістичної регресії; для дерев/бустингу — `class_weight` / `scale_pos_weight`
- композитний **рейтинг** (ROC-AUC 50% + Recall 30% + F1 20%) + PR-AUC
- гіперпараметричний тюнінг топ-2 моделей
- оптимальний поріг на validation (Youden) → default у UI
- зважений ensemble-підсумок; режим передбачення `all` | `best`
- веб-форма зі слайдером порогу та історією куріння
- **Flask** UI (Waitress, CSRF) + API: `/health`, `/api/predict`, `/api/explain`
- Docker (non-root), CI (GitHub Actions), unit-тести (`pytest`)

## Структура

```
ProjectAILearning/
├── app.py                      # Flask UI + REST API
├── train_diabetes_model.py     # навчання (CLI: --no-tune, --models, --sample)
├── predict_diabetes.py         # передбачення (mode=all|best)
├── model_registry.py           # реєстр алгоритмів / pipelines
├── scoring.py                  # єдиний selection_score
├── bootstrap_models.py         # cold-start навчання
├── explainability.py           # важливість ознак для /api/explain
├── validators.py / config.py / exceptions.py
├── diabetes_models.joblib      # повний пакет моделей (деплой)
├── .github/workflows/ci.yml
├── Dockerfile / docker-compose.yml
├── templates/ / static/
├── tests/
└── requirements.txt
```

## Вимоги

- Python 3.10+
- залежності з `requirements.txt`

## Швидкий старт (локально)

```bash
git clone https://github.com/prostir06/ProjectAILearning.git
cd ProjectAILearning

python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate

pip install -r requirements.txt
```

### Flask UI

```bash
python app.py
```

Відкрийте [http://127.0.0.1:5000](http://127.0.0.1:5000).

Якщо `diabetes_models.joblib` відсутній, додаток швидко навчить моделі без тюнінгу
(`QUICK_TRAIN_MAX_ROWS`, за замовчуванням 20000). Для продакшену комітьте готовий joblib.

- Health: `GET /health`
- API: `POST /api/predict`, `GET /api/explain`
- Debug: `FLASK_DEBUG=1`
- Секрет: `FLASK_SECRET_KEY`

### Навчання моделей

```bash
python train_diabetes_model.py
python train_diabetes_model.py --no-tune --sample 20000
python train_diabetes_model.py --models rf,xgb --serve-best-only
```

### Тести

```bash
python -m pytest tests/ -v
```

## Docker

```bash
docker compose up --build
# → http://localhost:5000
```

Або:

```bash
docker build -t diabetes-prediction .
docker run --rm -p 5000:5000 -e FLASK_SECRET_KEY=change-me diabetes-prediction
```

Публічний репозиторій: https://github.com/prostir06/ProjectAILearning

## Ліцензія даних

Датасет `diabetes_prediction_dataset.csv` — публічний навчальний набір. Перед комерційним використанням перевірте умови ліцензії оригінального джерела.
