"""
Unit-тести для ui_helpers.py.
"""

import ui_helpers


def test_clamp_percent_invalid_uses_default():
    """Нечислове значення → default."""
    assert ui_helpers.clamp_percent("x", default=42) == 42
    assert ui_helpers.clamp_percent(None, default=7) == 7


def test_build_results_grid_skips_non_dict_items():
    """Некоректні елементи списку не ламають сітку."""
    html = ui_helpers.build_results_grid_html(
        [
            "bad",
            {
                "model_name": "RF",
                "probability": 0.2,
                "diabetes": 0,
                "label": "Ні",
            },
        ],
        threshold_percent=50,
    )
    assert "st-results-grid" in html
    assert "RF" in html


def test_build_summary_block_handles_non_dict_summary():
    """summary=None не падає."""
    html = ui_helpers.build_summary_block_html(
        None,  # type: ignore[arg-type]
        threshold_percent=50,
        summary_percent=10,
        summary_positive=False,
    )
    assert "st-summary-block" in html
    assert "—" in html
