"""
Streamlit веб-інтерфейс для передбачення діабету.

Точка входу для Streamlit Community Cloud:
    streamlit run streamlit_app.py
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

import bootstrap_models
from app import format_metrics_for_display, get_error_message
from config import (
    DEFAULT_FORM,
    DEFAULT_THRESHOLD_PERCENT,
    MODELS_BUNDLE_PATH,
    SMOKING_OPTIONS_UK,
    THRESHOLD_MAX,
    THRESHOLD_MIN,
    THRESHOLD_STEP_PERCENT,
    VALID_RANGES,
)
from exceptions import (
    InvalidPatientDataError,
    ModelNotFoundError,
    PredictionError,
)
from predict_diabetes import (
    get_feature_importance,
    get_bundle_optimal_threshold,
    get_training_metrics,
    predict_with_summary,
)
from ui_helpers import (
    build_donut_html,
    build_model_card_html,
    build_results_grid_html,
    build_summary_block_html,
)
from validators import validate_person_data

# Стилі donut-діаграм (узгоджено з static/style.css у Flask UI).
DONUT_CHART_STYLES = """
<style>
.st-donut-wrap {
  display: flex;
  justify-content: center;
  margin: 0.75rem 0;
}
.st-donut {
  --percent: 0;
  --threshold: 50;
  --fill: #2dd4bf;
  position: relative;
  width: var(--size);
  height: var(--size);
  border-radius: 50%;
  background: conic-gradient(
    from 180deg,
    var(--fill) 0%,
    var(--fill) calc(var(--percent) * 1%),
    #1e293b calc(var(--percent) * 1%),
    #1e293b 100%
  );
  display: flex;
  align-items: center;
  justify-content: center;
  box-shadow: 0 4px 16px rgba(30, 41, 59, 0.1);
}
.st-donut-positive { --fill: #f97316; }
.st-donut-threshold {
  position: absolute;
  inset: 0;
  width: 100%;
  height: 100%;
  z-index: 2;
  pointer-events: none;
}
.st-donut-threshold line {
  stroke: #dc2626;
  stroke-width: 2;
  stroke-linecap: round;
}
.st-donut-hole {
  position: relative;
  z-index: 3;
  width: var(--hole);
  height: var(--hole);
  background: #fff;
  border-radius: 50%;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 0.15rem;
}
.st-donut-value {
  font-size: var(--value-size);
  font-weight: 700;
  color: #0f172a;
  line-height: 1;
}
.st-donut-label {
  font-size: var(--label-size);
  color: #334155;
  text-align: center;
}
.st-result-label {
  text-align: center;
  font-size: 1.5rem;
  font-weight: 700;
  margin: 0.25rem 0 0.5rem;
}
.st-result-negative { color: #15803d; }
.st-result-positive { color: #c2410c; }
.st-model-card {
  display: flex;
  flex-direction: column;
  align-items: center;
  text-align: center;
  padding: 1rem 0.75rem;
  border-radius: 12px;
  height: 100%;
  box-sizing: border-box;
}
.st-model-card-negative {
  background: #eef6f8;
  border: 1px solid #cfe8ee;
}
.st-model-card-positive {
  background: #fff7ed;
  border: 1px solid #fed7aa;
}
.st-results-grid {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 1rem;
  align-items: stretch;
  margin-top: 0.5rem;
}
.st-model-card-title {
  min-height: 4.5rem;
  display: flex;
  align-items: center;
  justify-content: center;
  width: 100%;
  line-height: 1.3;
  margin-bottom: 0.5rem;
}
.st-model-card-chart {
  display: flex;
  justify-content: center;
  align-items: center;
  width: 100%;
  min-height: 140px;
}
.st-model-result {
  font-size: 1.25rem;
  font-weight: 700;
  margin: 0.35rem 0 0.15rem;
  text-align: center;
}
.st-model-error {
  font-size: 0.85rem;
  color: #64748b;
  text-align: center;
  min-height: 1.25rem;
  margin: 0;
}
.st-summary-block {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
}
.st-donut-wrap.compact {
  margin: 0;
}
@media (max-width: 900px) {
  .st-results-grid {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }
}
@media (max-width: 560px) {
  .st-results-grid {
    grid-template-columns: 1fr;
  }
}
</style>
"""

# Метадані сторінки Streamlit Cloud.
st.set_page_config(
    page_title="Передбачення діабету",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource(show_spinner="Завантаження моделей…")
def ensure_models_ready() -> bool:
    """Гарантує наявність пакета моделей."""
    if MODELS_BUNDLE_PATH.exists():
        return True
    try:
        return bootstrap_models.ensure_models_ready()
    except RuntimeError as exc:
        if "першому запуску" in str(exc):
            raise
        raise RuntimeError(
            f"Не вдалося навчити моделі при першому запуску: {exc}"
        ) from exc


@st.cache_data(show_spinner=False)
def load_metrics_table() -> pd.DataFrame:
    """
    Метрики алгоритмів для таблиці порівняння.

    Returns:
        DataFrame з відсотковими колонками або порожній DataFrame
        при відсутності / пошкодженні метрик.
    """
    try:
        rows = format_metrics_for_display(get_training_metrics())
    except Exception:
        # Кешована функція не повинна падати через битий JSON / joblib.
        return pd.DataFrame()

    if not rows:
        return pd.DataFrame()

    try:
        table = pd.DataFrame(rows)
        formatted = {
            "#": table["rank"],
            "Алгоритм": table["model_name"],
            "Рейтинг %": (table["selection_score"] * 100).round(1),
            "ROC-AUC %": (table["roc_auc"] * 100).round(1),
            "Recall %": (table["recall"] * 100).round(1),
            "F1 %": (table["f1"] * 100).round(1),
            "Точність %": (table["accuracy"] * 100).round(1),
            "Похибка %": (table["error_rate"] * 100).round(1),
            "Найкраща": table["is_best"].map(
                lambda value: "так" if value else ""
            ),
            "Тюнінг": table["tuned"].map(
                lambda value: "так" if value else ""
            ),
        }
        if "pr_auc" in table.columns and table["pr_auc"].notna().any():
            formatted["PR-AUC %"] = (table["pr_auc"] * 100).round(1)
        return pd.DataFrame(formatted)
    except (KeyError, TypeError, ValueError):
        return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_importance_table() -> pd.DataFrame:
    """
    Важливість ознак найкращої моделі.

    Returns:
        DataFrame з колонками «Ознака» / «Важливість %»
        або порожній DataFrame при помилці читання.
    """
    try:
        items = get_feature_importance()
    except Exception:
        return pd.DataFrame()

    if not items:
        return pd.DataFrame()

    try:
        frame = pd.DataFrame(items)
        return pd.DataFrame({
            "Ознака": frame["label_uk"],
            "Важливість %": (frame["importance"] * 100).round(1),
        })
    except (KeyError, TypeError, ValueError):
        return pd.DataFrame()


def render_sidebar_form() -> tuple[dict | None, float, bool]:
    """
    Форма введення даних пацієнта в бічній панелі.

    Returns:
        Кортеж (person_dict або None, threshold 0–1, чи натиснуто кнопку).
    """
    st.sidebar.header("Дані пацієнта")
    st.sidebar.caption("Введіть показники для передбачення")

    gender_label = st.sidebar.selectbox(
        "Стать",
        options=["Жінка", "Чоловік"],
        index=0 if DEFAULT_FORM["gender"] == "Female" else 1,
    )
    gender = "Female" if gender_label == "Жінка" else "Male"

    age_min, age_max = VALID_RANGES["age"]
    age = st.sidebar.number_input(
        "Вік",
        min_value=int(age_min),
        max_value=int(age_max),
        value=int(float(DEFAULT_FORM["age"])),
        step=1,
    )

    hypertension = st.sidebar.selectbox(
        "Гіпертонія",
        options=[("Ні", 0), ("Так", 1)],
        format_func=lambda item: item[0],
        index=int(DEFAULT_FORM["hypertension"]),
    )[1]

    heart_disease = st.sidebar.selectbox(
        "Хвороби серця",
        options=[("Ні", 0), ("Так", 1)],
        format_func=lambda item: item[0],
        index=int(DEFAULT_FORM["heart_disease"]),
    )[1]

    smoking_items = list(SMOKING_OPTIONS_UK.items())
    default_smoking = str(DEFAULT_FORM.get("smoking_history", "No Info"))
    default_smoking_index = next(
        (
            index
            for index, (value, _) in enumerate(smoking_items)
            if value == default_smoking
        ),
        len(smoking_items) - 1,
    )
    smoking_history = st.sidebar.selectbox(
        "Історія куріння",
        options=smoking_items,
        format_func=lambda item: item[1],
        index=default_smoking_index,
    )[0]

    bmi_min, bmi_max = VALID_RANGES["bmi"]
    bmi = st.sidebar.number_input(
        "ІМТ (індекс маси тіла)",
        min_value=float(bmi_min),
        max_value=float(bmi_max),
        value=float(DEFAULT_FORM["bmi"]),
        step=0.1,
        format="%.1f",
    )

    hba1c_min, hba1c_max = VALID_RANGES["HbA1c_level"]
    hba1c = st.sidebar.number_input(
        "HbA1c (%)",
        min_value=float(hba1c_min),
        max_value=float(hba1c_max),
        value=float(DEFAULT_FORM["HbA1c_level"]),
        step=0.1,
        format="%.1f",
        help="Середній показник цукру в крові за 2–3 місяці",
    )

    glucose_min, glucose_max = VALID_RANGES["blood_glucose_level"]
    glucose = st.sidebar.number_input(
        "Глюкоза в крові (мг/дл)",
        min_value=int(glucose_min),
        max_value=int(glucose_max),
        value=int(DEFAULT_FORM["blood_glucose_level"]),
        step=1,
    )

    default_threshold_percent = int(round(
        get_bundle_optimal_threshold(
            default=DEFAULT_THRESHOLD_PERCENT / 100.0
        ) * 100
    ))
    threshold_percent = st.sidebar.slider(
        "Поріг ймовірності (%)",
        min_value=int(THRESHOLD_MIN * 100),
        max_value=int(THRESHOLD_MAX * 100),
        value=default_threshold_percent,
        step=THRESHOLD_STEP_PERCENT,
        help=(
            "Якщо ймовірність ≥ порогу — результат «Так». "
            "Нижчий поріг — більше позитивних відповідей."
        ),
    )

    submitted = st.sidebar.button("Передбачити", type="primary", use_container_width=True)

    person = {
        "gender": gender,
        "age": age,
        "hypertension": hypertension,
        "heart_disease": heart_disease,
        "smoking_history": smoking_history,
        "bmi": bmi,
        "HbA1c_level": hba1c,
        "blood_glucose_level": glucose,
    }
    return person, threshold_percent / 100.0, submitted


def render_metrics_section() -> None:
    """Таблиця метрик і важливість ознак."""
    st.subheader("Похибка алгоритмів на тестовій вибірці")
    st.caption(
        "SMOTE на train, метрики на test (80% / 20%). "
        "Сортування за рейтингом (ROC-AUC 50% + Recall 30% + F1 20%)."
    )

    metrics_table = load_metrics_table()
    if metrics_table.empty:
        st.info("Метрики ще не збережені. Запустіть `python train_diabetes_model.py`.")
    else:
        st.dataframe(metrics_table, use_container_width=True, hide_index=True)

    importance = load_importance_table()
    if not importance.empty:
        st.subheader("Важливість ознак (найкраща модель)")
        st.bar_chart(importance.set_index("Ознака")["Важливість %"])


def render_prediction(person: dict, threshold: float) -> None:
    """
    Виконує передбачення та показує підсумок + картки моделей.

    Args:
        person: Нормалізовані або сирі дані пацієнта з форми.
        threshold: Поріг ймовірності в діапазоні 0.0–1.0.
    """
    try:
        validate_person_data(person)
        prediction = predict_with_summary(person, threshold=threshold)
    except (InvalidPatientDataError, ModelNotFoundError, PredictionError) as exc:
        st.error(get_error_message(exc))
        return
    except Exception as exc:
        # Несподівані збої (наприклад, несумісна версія sklearn).
        st.error(get_error_message(exc))
        return

    try:
        summary = prediction["summary"]
        models = prediction["models"]
        threshold_percent = int(threshold * 100)
        summary_percent = int(round(float(summary["probability"]) * 100))
        summary_positive = int(summary["diabetes"]) == 1
    except (KeyError, TypeError, ValueError) as exc:
        st.error(get_error_message(PredictionError(str(exc))))
        return

    st.subheader("Загальний підсумок")
    st.markdown(
        build_summary_block_html(
            summary,
            threshold_percent,
            summary_percent,
            summary_positive,
        ),
        unsafe_allow_html=True,
    )
    st.caption(summary.get("votes_text", ""))
    st.caption(f"Поріг: {threshold_percent}% (вище = «Так», відлік знизу)")

    st.subheader("Результати за алгоритмами")
    st.caption(
        f"Червона лінія — поріг {threshold_percent}% "
        "(вище = «Так», відлік знизу)"
    )
    st.markdown(
        build_results_grid_html(models, threshold_percent),
        unsafe_allow_html=True,
    )


def main() -> None:
    """Головна сторінка Streamlit-додатка."""
    st.markdown(DONUT_CHART_STYLES, unsafe_allow_html=True)
    st.title("Передбачення діабету")
    st.markdown(
        "Порівняння кількох алгоритмів ML за даними пацієнта. "
        "**Навчальна модель — не замінює медичну діагностику.**"
    )

    try:
        ensure_models_ready()
    except Exception as exc:
        st.error(
            "Не вдалося підготувати моделі. "
            f"Деталі: {exc}"
        )
        st.stop()

    person, threshold, submitted = render_sidebar_form()

    left, right = st.columns([1.1, 1.0], gap="large")
    with left:
        render_metrics_section()
    with right:
        if submitted and person is not None:
            render_prediction(person, threshold)
        else:
            st.info(
                "Заповніть форму ліворуч (бічна панель) і натисніть «Передбачити»."
            )


if __name__ == "__main__":
    main()
