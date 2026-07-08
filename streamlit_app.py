"""
Streamlit веб-інтерфейс для передбачення діабету.

Точка входу для Streamlit Community Cloud:
    streamlit run streamlit_app.py
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app import format_metrics_for_display, get_error_message
from config import (
    DEFAULT_FORM,
    DEFAULT_THRESHOLD_PERCENT,
    MODELS_BUNDLE_PATH,
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
    get_training_metrics,
    predict_with_summary,
    reset_pipeline_cache,
)
from validators import validate_person_data

# Метадані сторінки Streamlit Cloud.
st.set_page_config(
    page_title="Передбачення діабету",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource(show_spinner="Завантаження моделей…")
def ensure_models_ready() -> bool:
    """
    Гарантує наявність пакета моделей.

    Якщо diabetes_models.joblib відсутній (локально без артефакту),
    навчає моделі без тюнінгу (прискорений шлях для першого запуску).

    Returns:
        True, якщо моделі доступні після перевірки / навчання.
    """
    if MODELS_BUNDLE_PATH.exists():
        return True

    from train_diabetes_model import (
        save_feature_importance,
        save_metrics_json,
        save_models_bundle,
        train_all_models,
    )

    models, metrics, best_key, importance = train_all_models(enable_tuning=False)
    save_models_bundle(models, metrics, best_key, importance)
    save_metrics_json(metrics)
    save_feature_importance(importance)
    reset_pipeline_cache()
    return MODELS_BUNDLE_PATH.exists()


@st.cache_data(show_spinner=False)
def load_metrics_table() -> pd.DataFrame:
    """Метрики алгоритмів для таблиці порівняння."""
    rows = format_metrics_for_display(get_training_metrics())
    if not rows:
        return pd.DataFrame()

    table = pd.DataFrame(rows)
    display = pd.DataFrame({
        "#": table["rank"],
        "Алгоритм": table["model_name"],
        "Рейтинг %": (table["selection_score"] * 100).round(1),
        "ROC-AUC %": (table["roc_auc"] * 100).round(1),
        "Recall %": (table["recall"] * 100).round(1),
        "F1 %": (table["f1"] * 100).round(1),
        "Точність %": (table["accuracy"] * 100).round(1),
        "Похибка %": (table["error_rate"] * 100).round(1),
        "Найкраща": table["is_best"].map(lambda value: "так" if value else ""),
        "Тюнінг": table["tuned"].map(lambda value: "так" if value else ""),
    })
    return display


@st.cache_data(show_spinner=False)
def load_importance_table() -> pd.DataFrame:
    """Важливість ознак найкращої моделі."""
    items = get_feature_importance()
    if not items:
        return pd.DataFrame()

    frame = pd.DataFrame(items)
    return pd.DataFrame({
        "Ознака": frame["label_uk"],
        "Важливість %": (frame["importance"] * 100).round(1),
    })


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

    threshold_percent = st.sidebar.slider(
        "Поріг ймовірності (%)",
        min_value=int(THRESHOLD_MIN * 100),
        max_value=int(THRESHOLD_MAX * 100),
        value=DEFAULT_THRESHOLD_PERCENT,
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
        "smoking_history": "No Info",
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
    """Виконує передбачення та показує підсумок + картки моделей."""
    try:
        validate_person_data(person)
        prediction = predict_with_summary(person, threshold=threshold)
    except (InvalidPatientDataError, ModelNotFoundError, PredictionError) as exc:
        st.error(get_error_message(exc))
        return
    except Exception as exc:
        st.error(get_error_message(exc))
        return

    summary = prediction["summary"]
    models = prediction["models"]
    threshold_percent = int(threshold * 100)

    st.subheader("Загальний підсумок")
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Результат", summary["label"])
    col_b.metric(
        "Середня ймовірність",
        f"{summary['probability'] * 100:.0f}%",
    )
    col_c.metric("Поріг", f"{threshold_percent}%")
    st.caption(summary["votes_text"])
    st.progress(min(1.0, max(0.0, float(summary["probability"]))))

    st.subheader("Результати за алгоритмами")
    st.caption(f"Червона лінія порогу: {threshold_percent}% (вище = «Так»)")

    columns = st.columns(3)
    for index, item in enumerate(models):
        with columns[index % 3]:
            title = item["model_name"]
            if item.get("rank"):
                title = f"#{item['rank']} {title}"
            if item.get("is_best"):
                title += " · найкраща"

            probability = float(item["probability"])
            st.markdown(f"**{title}**")
            st.metric("Ймовірність", f"{probability * 100:.0f}%", item["label"])
            st.progress(min(1.0, max(0.0, probability)))
            if item.get("error_rate") is not None:
                st.caption(f"Похибка на тесті: {item['error_rate'] * 100:.1f}%")
            st.divider()


def main() -> None:
    """Головна сторінка Streamlit-додатка."""
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
