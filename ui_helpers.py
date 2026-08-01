"""
Спільні HTML-хелпери для donut-діаграм і карток результатів.

Використовується Streamlit; Flask лишає Jinja, але логіка відсотків/escape спільна.
"""

from __future__ import annotations


def escape_html(text: object) -> str:
    """Екранує HTML-спецсимволи для безпечного вставлення в розмітку."""
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def clamp_percent(value: object, default: int = 0) -> int:
    """Обмежує значення відсотка діапазоном 0–100."""
    try:
        return max(0, min(100, int(value)))
    except (TypeError, ValueError):
        return default


def build_donut_html(
    percent: int,
    threshold_percent: int,
    donut_label: str,
    is_positive: bool,
    *,
    small: bool = False,
    compact: bool = False,
    css_prefix: str = "st-",
) -> str:
    """
    Генерує HTML donut-діаграми з червоною лінією порогу.

    Відлік дуги й порогу починається знизу (6 год.) за годинниковою стрілкою.

    Args:
        percent: Ймовірність у відсотках (0–100).
        threshold_percent: Поріг у відсотках (0–100).
        donut_label: Підпис у центрі діаграми.
        is_positive: Чи результат «Так» (помаранчева дуга).
        small: Менший розмір для карток алгоритмів.
        compact: Без зовнішніх відступів (всередині картки).
        css_prefix: Префікс CSS-класів (``st-`` для Streamlit).

    Returns:
        HTML-рядок.
    """
    percent = clamp_percent(percent)
    threshold_percent = clamp_percent(threshold_percent, default=50)

    size = 140 if small else 180
    hole = 100 if small else 132
    value_size = "1.75rem" if small else "2.25rem"
    label_size = "0.85rem" if small else "0.95rem"
    positive_class = f" {css_prefix}donut-positive" if is_positive else ""
    threshold_rotation = threshold_percent * 3.6
    wrap_class = (
        f"{css_prefix}donut-wrap compact"
        if compact
        else f"{css_prefix}donut-wrap"
    )
    safe_label = escape_html(donut_label)

    return f"""
<div class="{wrap_class}">
  <div
    class="{css_prefix}donut{positive_class}"
    style="--percent: {percent}; --threshold: {threshold_percent};
           --size: {size}px; --hole: {hole}px;
           --value-size: {value_size}; --label-size: {label_size};"
    role="img"
    aria-label="Ймовірність {percent} відсотків, поріг {threshold_percent} відсотків"
  >
    <svg class="{css_prefix}donut-threshold" viewBox="0 0 100 100" aria-hidden="true">
      <line
        x1="50" y1="50" x2="50" y2="90"
        transform="rotate({threshold_rotation} 50 50)"
      />
    </svg>
    <div class="{css_prefix}donut-hole">
      <span class="{css_prefix}donut-value">{percent}%</span>
      <span class="{css_prefix}donut-label">{safe_label}</span>
    </div>
  </div>
</div>
"""


def build_model_card_html(item: dict, threshold_percent: int) -> str:
    """
    HTML однієї картки алгоритму з вирівняними блоками.

    Returns:
        HTML-картка; при некоректних даних — порожній рядок.
    """
    try:
        title = escape_html(item.get("model_name", "Модель"))
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
        label = escape_html(item.get("label", "—"))

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
    """Сітка карток алгоритмів 3×2 з однаковим вирівнюванням."""
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
    """HTML блоку загального підсумку з центрованою donut-діаграмою."""
    result_class = (
        "st-result-positive" if summary_positive else "st-result-negative"
    )
    label = escape_html(summary.get("label", "—"))
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
