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
    """
    Гарантує наявність пакета моделей.

    Якщо diabetes_models.joblib відсутній (локально без артефакту),
    навчає моделі без тюнінгу (прискорений шлях для першого запуску).

    Returns:
        True, якщо моделі доступні після перевірки / навчання.
    """
    if MODELS_BUNDLE_PATH.exists():
        return True

    # Локальний / Docker cold-start: навчаємо без тюнінгу, щоб UI стартував швидше.
    from train_diabetes_model import (
        save_feature_importance,
        save_metrics_json,
        save_models_bundle,
        train_all_models,
    )
    from exceptions import DataLoadError

    try:
        models, metrics, best_key, importance = train_all_models(
            enable_tuning=False,
        )
        save_models_bundle(models, metrics, best_key, importance)
        save_metrics_json(metrics)
        save_feature_importance(importance)
    except (DataLoadError, OSError, ValueError) as exc:
        # Передаємо далі — main() покаже повідомлення користувачу.
        raise RuntimeError(
            f"Не вдалося навчити моделі при першому запуску: {exc}"
        ) from exc

    reset_pipeline_cache()
    return MODELS_BUNDLE_PATH.exists()


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
        return pd.DataFrame({
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
        })
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


def _escape_html(text: object) -> str:
    """Екранує HTML-спецсимволи для безпечного вставлення в розмітку."""
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def build_donut_html(
    percent: int,
    threshold_percent: int,
    donut_label: str,
    is_positive: bool,
    *,
    small: bool = False,
    compact: bool = False,
) -> str:
    """
    Генерує HTML donut-діаграми з червоною лінією порогу.

    Відлік дуги й порогу починається знизу (6 год.) за годинниковою стрілкою,
    як у Flask-інтерфейсі.

    Args:
        percent: Ймовірність у відсотках (0–100).
        threshold_percent: Поріг у відсотках (0–100).
        donut_label: Підпис у центрі діаграми.
        is_positive: Чи результат «Так» (помаранчева дуга).
        small: Менший розмір для карток алгоритмів.
        compact: Без зовнішніх відступів (всередині картки).

    Returns:
        HTML-рядок для st.markdown(..., unsafe_allow_html=True).
    """
    # Обмежуємо відсотки, щоб CSS conic-gradient і SVG не ламались.
    try:
        percent = max(0, min(100, int(percent)))
        threshold_percent = max(0, min(100, int(threshold_percent)))
    except (TypeError, ValueError):
        percent = 0
        threshold_percent = 50

    size = 140 if small else 180
    hole = 100 if small else 132
    value_size = "1.75rem" if small else "2.25rem"
    label_size = "0.85rem" if small else "0.95rem"
    positive_class = " st-donut-positive" if is_positive else ""
    threshold_rotation = threshold_percent * 3.6
    wrap_class = "st-donut-wrap compact" if compact else "st-donut-wrap"
    safe_label = _escape_html(donut_label)

    return f"""
<div class="{wrap_class}">
  <div
    class="st-donut{positive_class}"
    style="--percent: {percent}; --threshold: {threshold_percent};
           --size: {size}px; --hole: {hole}px;
           --value-size: {value_size}; --label-size: {label_size};"
    role="img"
    aria-label="Ймовірність {percent} відсотків, поріг {threshold_percent} відсотків"
  >
    <svg class="st-donut-threshold" viewBox="0 0 100 100" aria-hidden="true">
      <line
        x1="50" y1="50" x2="50" y2="90"
        transform="rotate({threshold_rotation} 50 50)"
      />
    </svg>
    <div class="st-donut-hole">
      <span class="st-donut-value">{percent}%</span>
      <span class="st-donut-label">{safe_label}</span>
    </div>
  </div>
</div>
"""


def build_model_card_html(item: dict, threshold_percent: int) -> str:
    """
    HTML однієї картки алгоритму з вирівняними блоками.

    Args:
        item: Результат одного алгоритму (model_name, probability, …).
        threshold_percent: Поріг у відсотках для donut-лінії.

    Returns:
        HTML-картка; при некоректних даних — порожній рядок.
    """
    try:
        title = _escape_html(item.get("model_name", "Модель"))
        if item.get("rank"):
            title = f"#{int(item['rank'])} {title}"
        if item.get("is_best"):
            title += " · найкраща"

        percent = int(round(float(item["probability"]) * 100))
        is_positive = int(item["diabetes"]) == 1
        card_class = (
            "st-model-card-positive" if is_positive else "st-model-card-negative"
        )
        result_class = (
            "st-result-positive" if is_positive else "st-result-negative"
        )
        label = _escape_html(item.get("label", "—"))

        error_text = ""
        if item.get("error_rate") is not None:
            error_text = (
                f"Похибка на тесті: {float(item['error_rate']) * 100:.1f}%"
            )

        donut = build_donut_html(
            percent,
            threshold_percent,
            "ймовірність",
            is_positive,
            small=True,
            compact=True,
        )
    except (KeyError, TypeError, ValueError):
        return ""

    return f"""
<div class="st-model-card {card_class}">
  <div class="st-model-card-title"><strong>{title}</strong></div>
  <div class="st-model-card-chart">{donut}</div>
  <p class="st-model-result {result_class}">{label}</p>
  <p class="st-model-error">{error_text}</p>
</div>
"""


def build_results_grid_html(models: list[dict], threshold_percent: int) -> str:
    """
    Сітка карток алгоритмів 3×2 з однаковим вирівнюванням.

    Args:
        models: Список результатів predict_with_summary()["models"].
        threshold_percent: Поріг у відсотках.

    Returns:
        HTML-контейнер .st-results-grid.
    """
    if not models:
        return '<div class="st-results-grid"></div>'

    cards = "".join(
        build_model_card_html(item, threshold_percent) for item in models
    )
    return f'<div class="st-results-grid">{cards}</div>'


def build_summary_block_html(
    summary: dict,
    threshold_percent: int,
    summary_percent: int,
    summary_positive: bool,
) -> str:
    """
    HTML блоку загального підсумку з центрованою donut-діаграмою.

    Args:
        summary: Словник підсумку (label, probability, …).
        threshold_percent: Поріг у відсотках.
        summary_percent: Середня ймовірність у відсотках.
        summary_positive: True, якщо підсумок «Так».

    Returns:
        HTML блоку .st-summary-block.
    """
    result_class = (
        "st-result-positive" if summary_positive else "st-result-negative"
    )
    label = _escape_html(summary.get("label", "—"))
    donut = build_donut_html(
        summary_percent,
        threshold_percent,
        "середня ймовірність",
        summary_positive,
    )
    return f"""
<div class="st-summary-block">
  {donut}
  <p class="st-result-label {result_class}">{label}</p>
</div>
"""


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
