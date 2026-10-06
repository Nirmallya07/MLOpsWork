import pytest

from orders import minutes_per_km


def test_slow_delivery():
    assert minutes_per_km(40, 10) > minutes_per_km(30, 10)


def test_negative_distance_raises():
    with pytest.raises(ValueError):
        minutes_per_km(30, -5)
