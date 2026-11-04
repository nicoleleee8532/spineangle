import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cobb import (angle_between, aasce_angles, curves_from_landmarks,  # noqa: E402
                  endplate_tilts, manual_cobb, severity_band, smape, tilt_deg)
from synthetic import synthetic_landmarks  # noqa: E402


def test_tilt_is_normalised_regardless_of_point_order():
    a, b = np.array([0, 0]), np.array([10, 10])
    assert tilt_deg(a, b) == pytest.approx(45)
    assert tilt_deg(b, a) == pytest.approx(45)  # reversed order, same line


def test_angle_between_wraps_near_vertical():
    assert angle_between(89, -89) == pytest.approx(2)


def test_manual_cobb_simple():
    # upper endplate tilted -20 deg, lower +15 deg -> 35 deg
    up = [(0, 0), (np.cos(np.radians(-20)), np.sin(np.radians(-20)))]
    lo = [(0, 0), (np.cos(np.radians(15)), np.sin(np.radians(15)))]
    assert manual_cobb(*up, *lo) == pytest.approx(35)


def test_straight_spine_is_zero():
    lm = synthetic_landmarks(0.0)
    assert curves_from_landmarks(lm)[0].cobb_deg == pytest.approx(0, abs=1e-6)


@pytest.mark.parametrize("angle", [12, 25, 38, 52, 70])
def test_synthetic_curve_recovers_known_angle(angle):
    lm = synthetic_landmarks(angle)
    primary = curves_from_landmarks(lm)[0]
    assert primary.cobb_deg == pytest.approx(angle, abs=0.01)
    assert primary.upper_vertebra == "T5"   # vertebra index 4
    assert primary.lower_vertebra == "L2"   # vertebra index 13


def test_noise_keeps_angle_close():
    lm = synthetic_landmarks(40, noise_px=1.0, seed=0)
    assert abs(curves_from_landmarks(lm)[0].cobb_deg - 40) < 5


def test_endplate_count():
    assert endplate_tilts(synthetic_landmarks(30)).shape == (34,)


def test_aasce_angles_primary_in_middle_slot():
    out = aasce_angles(synthetic_landmarks(30))
    assert out[1] == pytest.approx(30, abs=0.01)


def test_smape_perfect_is_zero():
    t = np.array([[10, 40, 20.0]])
    assert smape(t, t) == 0


def test_severity_bands():
    assert severity_band(5) == "below scoliosis range"
    assert severity_band(18) == "mild"
    assert severity_band(30) == "moderate"
    assert severity_band(55) == "severe"


def test_js_example_matches_python():
    """The web page's example spine is built for 38 deg; Python must agree."""
    assert curves_from_landmarks(synthetic_landmarks(38))[0].cobb_deg == pytest.approx(38, abs=0.01)
