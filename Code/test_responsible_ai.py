"""Automated test suite — run with:
    pytest test_responsible_ai.py -v --cov=responsible_ai_utils --cov-report=term
"""
import numpy as np
import pandas as pd
import pytest

from responsible_ai_utils import (age_to_group, parse_utk_filename, clean_utk,
                                  splits_disjoint, one_off_correct,
                                  accuracy_parity_gap, fairness_check,
                                  p_at_least, decide,
                                  AGE_GROUP_LABELS)


# ---------- age_to_group: every boundary value ----------
@pytest.mark.parametrize("age,expected", [
    (1,  "0-2"),  (2,  "0-2"),
    (3,  "3-9"),  (9,  "3-9"),
    (10, "10-19"), (19, "10-19"),
    (20, "20-29"), (29, "20-29"),
    (30, "30-39"), (39, "30-39"),
    (40, "40-49"), (49, "40-49"),
    (50, "50-59"), (59, "50-59"),
    (60, "60-69"), (69, "60-69"),
    (70, "70+"),   (100, "70+"),
])
def test_age_to_group_boundaries(age, expected):
    assert age_to_group(age) == expected


def test_age_groups_are_canonical():
    # every produced label must be one of the 9 canonical labels
    for age in range(1, 101):
        assert age_to_group(age) in AGE_GROUP_LABELS


# ---------- filename parsing ----------
def test_parse_valid_filename():
    r = parse_utk_filename("25_0_2_20170116174525125")
    assert r == {"age": 25, "gender": "Male", "race": "Asian"}

def test_parse_female_white():
    r = parse_utk_filename("61_1_0_20170109150557335")
    assert r == {"age": 61, "gender": "Female", "race": "White"}

def test_parse_too_few_parts_returns_none():
    assert parse_utk_filename("25_0") is None

def test_parse_non_numeric_returns_none():
    assert parse_utk_filename("abc_0_1_xyz") is None

def test_parse_unknown_codes_mapped():
    r = parse_utk_filename("30_7_9_ts")           # invalid gender/race codes
    assert r["gender"] == "Unknown" and r["race"] == "Unknown"


# ---------- cleaning rules ----------
def _toy_df():
    return pd.DataFrame({
        "filepath": [f"f{i}.jpg" for i in range(6)],
        "age"     : [0, 1, 50, 100, 116, 30],
        "gender"  : ["Male", "Female", "Male", "Female", "Male", "Unknown"],
        "race"    : ["White", "Black", "Unknown", "Asian", "Indian", "Other"],
    })

def test_clean_removes_age_zero_and_over_100():
    out = clean_utk(_toy_df())
    assert out["age"].min() >= 1
    assert out["age"].max() <= 100

def test_clean_removes_unknown_labels():
    out = clean_utk(_toy_df())
    assert "Unknown" not in out["gender"].values
    assert "Unknown" not in out["race"].values

def test_clean_adds_valid_age_group_column():
    out = clean_utk(_toy_df())
    assert "age_group" in out.columns
    assert out["age_group"].isin(AGE_GROUP_LABELS).all()

def test_clean_keeps_expected_rows():
    # rows 1 (age 1) and 3 (age 100) survive; 0, 2, 4, 5 are filtered
    out = clean_utk(_toy_df())
    assert set(out["filepath"]) == {"f1.jpg", "f3.jpg"}


# ---------- split integrity ----------
def test_splits_disjoint_true_for_clean_splits():
    a = pd.DataFrame({"filepath": ["a.jpg", "b.jpg"]})
    b = pd.DataFrame({"filepath": ["c.jpg"]})
    c = pd.DataFrame({"filepath": ["d.jpg", "e.jpg"]})
    assert splits_disjoint(a, b, c)

def test_splits_disjoint_detects_leakage():
    a = pd.DataFrame({"filepath": ["a.jpg", "b.jpg"]})
    b = pd.DataFrame({"filepath": ["b.jpg"]})      # leaked into validation
    assert not splits_disjoint(a, b)


# ---------- metrics ----------
def test_one_off_correct():
    y_true = [3, 3, 3, 3]
    y_pred = [3, 4, 2, 6]        # exact, +1, -1, far
    assert one_off_correct(y_true, y_pred) == pytest.approx(0.75)

def test_accuracy_parity_gap():
    assert accuracy_parity_gap({"A": 0.55, "B": 0.42}) == pytest.approx(0.13)

def test_fairness_check_passes_within_threshold():
    assert fairness_check({"A": 0.50, "B": 0.45}, max_gap=0.10)

def test_fairness_check_fails_beyond_threshold():
    assert not fairness_check({"A": 0.56, "B": 0.42}, max_gap=0.10)


# ---------- deployment decision logic (age-restricted sales) ----------
def _probs(**kw):
    """Build a 9-band probability vector, e.g. _probs(**{'0': .8, '2': .2})."""
    p = [0.0] * 9
    for i, v in kw.items():
        p[int(i)] = v
    return p

def test_all_adult_mass_gives_certainty():
    assert p_at_least(_probs(**{"4": 1.0}), band_frac=0.2) == pytest.approx(1.0)

def test_all_child_mass_gives_zero():
    assert p_at_least(_probs(**{"0": 1.0}), band_frac=0.2) == pytest.approx(0.0)

def test_teen_band_counts_only_partially():
    # all mass in 10-19, 20% of that band is 18+  ->  P = 0.20
    assert p_at_least(_probs(**{"2": 1.0}), band_frac=0.2) == pytest.approx(0.2)

def test_probability_stays_in_range():
    for p in [_probs(**{"0": 1.0}), _probs(**{"2": 1.0}), _probs(**{"8": 1.0})]:
        assert 0.0 <= p_at_least(p, 0.2) <= 1.0

def test_lower_threshold_is_never_stricter():
    # more of the 10-19 band is 16+ than 18+, so P(>=16) >= P(>=18)
    p = _probs(**{"2": 1.0})
    assert p_at_least(p, band_frac=0.40) >= p_at_least(p, band_frac=0.20)

def test_decision_thresholds():
    assert decide(0.99) == "auto-clear"
    assert decide(0.01) == "auto-reject"
    assert decide(0.50) == "human-review"
    assert decide(0.95) == "auto-clear"           # boundary is inclusive
    assert decide(0.05) == "auto-reject"


# ---------- SAFETY: minors must never be auto-approved ----------
def test_young_child_is_never_auto_cleared():
    for band in ["0", "1"]:                       # 0-2 and 3-9
        assert decide(p_at_least(_probs(**{band: 1.0}), 0.2)) != "auto-clear"

def test_teen_band_alone_never_auto_clears():
    # 10-19 straddles the legal age of 18: on its own it must always escalate
    assert decide(p_at_least(_probs(**{"2": 1.0}), 0.2)) != "auto-clear"

def test_mixed_child_teen_never_auto_clears():
    p = _probs(**{"1": 0.5, "2": 0.5})            # half 3-9, half 10-19
    assert decide(p_at_least(p, 0.2)) != "auto-clear"