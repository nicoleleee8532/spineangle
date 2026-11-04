import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cobb import tilt_deg  # noqa: E402
from pattern import (SoftmaxRegression, apex_and_side, classify_pattern, features,  # noqa: E402
                     marks_from_landmarks, region_of_apex, synthetic_dataset, synthetic_spine)


def test_region_boundaries_follow_srs():
    assert region_of_apex(1) == "thoracic"      # T2
    assert region_of_apex(10) == "thoracic"     # T11
    assert region_of_apex(11) == "thoracolumbar"  # T12
    assert region_of_apex(12) == "thoracolumbar"  # L1
    assert region_of_apex(13) == "lumbar"       # L2


def test_single_thoracic_curve_to_image_right():
    lm = synthetic_spine([(3, 11, 40, +1)])
    out = classify_pattern(lm)
    assert out["pattern"] == "single thoracic"
    c = out["curves"][0]
    assert c["side"] == "image-right"
    assert c["cobb_deg"] == pytest.approx(40, abs=0.1)


def test_s_curve_is_double_with_opposite_sides():
    lm = synthetic_spine([(2, 8, 42, +1), (8, 15, 30, -1)])
    out = classify_pattern(lm)
    assert out["pattern"] == "double (thoracic and thoracolumbar/lumbar)"
    sides = {c["region"]: c["side"] for c in out["curves"] if c["major"]}
    assert sides["thoracic"] == "image-right"
    assert sides[next(r for r in sides if r != "thoracic")] == "image-left"


def test_side_rule_from_taps_matches_apex_side():
    """The browser infers side from the two tapped lines: upper tilt < lower tilt means image-right."""
    for sign in (+1, -1):
        lm = synthetic_spine([(3, 11, 35, sign)])
        m = marks_from_landmarks(lm, 6, 23)
        from_taps = "image-right" if tilt_deg(m[0], m[1]) < tilt_deg(m[2], m[3]) else "image-left"
        assert from_taps == apex_and_side(lm, 6, 23)[1]


def test_small_compensatory_curves_are_minor():
    out = classify_pattern(synthetic_spine([(3, 11, 40, +1)]))
    assert sum(c["major"] for c in out["curves"]) == 1
    assert len(out["curves"]) > 1          # compensations are still listed


def test_straight_spine_has_no_pattern():
    assert classify_pattern(synthetic_spine([]))["pattern"] == "none"


def test_feature_vector_shape():
    lm = synthetic_spine([(3, 11, 35, 1)])
    assert features(marks_from_landmarks(lm, 6, 23), 1100, 600).shape == (7,)


def test_model_beats_chance_and_mostly_confuses_neighbours():
    X, y = synthetic_dataset(1500, seed=1)
    cut = int(0.8 * len(X))
    m = SoftmaxRegression(epochs=800).fit(X[:cut], y[:cut])
    pred = m.predict(X[cut:])
    assert (pred == y[cut:]).mean() > 0.8
    assert not np.any((y[cut:] == 0) & (pred == 2))   # never thoracic -> lumbar
