"""
Unit-тести для diabetes.web.forms (поріг API та стійкість).
"""

import pytest

from diabetes.core.exceptions import InvalidPatientDataError
from diabetes.web.forms import (
    parse_threshold_from_form,
    parse_threshold_from_payload,
)


def test_parse_threshold_from_payload_fraction():
    """Значення 0–1 трактується як частка."""
    assert parse_threshold_from_payload(0.3) == 0.3
    assert parse_threshold_from_payload("0.45") == 0.45


def test_parse_threshold_from_payload_percent():
    """Значення >1 трактується як відсотки."""
    assert parse_threshold_from_payload(30) == 0.3
    assert parse_threshold_from_payload("40") == 0.4


def test_parse_threshold_from_payload_none_uses_default():
    """None → default."""
    assert parse_threshold_from_payload(None, default=0.42) == 0.42


def test_parse_threshold_from_payload_invalid_raises():
    """Нечислове значення → InvalidPatientDataError."""
    with pytest.raises(InvalidPatientDataError, match="числом"):
        parse_threshold_from_payload("abc")


def test_parse_threshold_from_payload_out_of_range_raises():
    """Поріг поза діапазоном → InvalidPatientDataError."""
    with pytest.raises(InvalidPatientDataError, match="діапазоні"):
        parse_threshold_from_payload(0.05)


def test_parse_threshold_from_form_type_error_returns_default():
    """Битий form_data без __contains__ не падає."""
    class Broken:
        def __contains__(self, _key):
            raise TypeError("broken")

    assert parse_threshold_from_form(Broken(), default=0.5) == 0.5
