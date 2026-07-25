"""Pure pipeline logic extracted from the notebook so pytest can import and test it.
Kept in exact sync with the notebook definitions."""

AGE_GROUP_LABELS = ["0-2", "3-9", "10-19", "20-29", "30-39",
                    "40-49", "50-59", "60-69", "70+"]

RACE_MAP   = {0: "White", 1: "Black", 2: "Asian", 3: "Indian", 4: "Other"}
GENDER_MAP = {0: "Male",  1: "Female"}


def age_to_group(age):
    """Bin a continuous age into one of 9 canonical age groups."""
    if   age <= 2:  return "0-2"
    elif age <= 9:  return "3-9"
    elif age <= 19: return "10-19"
    elif age <= 29: return "20-29"
    elif age <= 39: return "30-39"
    elif age <= 49: return "40-49"
    elif age <= 59: return "50-59"
    elif age <= 69: return "60-69"
    else:           return "70+"


def parse_utk_filename(stem):
    """Parse a UTKFace filename stem 'age_gender_race_timestamp'.
    Returns a dict or None for malformed / non-numeric stems."""
    parts = stem.split("_")
    if len(parts) < 3:
        return None
    try:
        return {
            "age"    : int(parts[0]),
            "gender" : GENDER_MAP.get(int(parts[1]), "Unknown"),
            "race"   : RACE_MAP.get(int(parts[2]),   "Unknown"),
        }
    except ValueError:
        return None


def clean_utk(df):
    """Apply the notebook's cleaning rules: age 1-100, no Unknown labels,
    add the age_group column."""
    out = df.copy()
    out = out[out["age"] >= 1]
    out = out[out["age"] <= 100]
    out = out[out["gender"] != "Unknown"]
    out = out[out["race"]   != "Unknown"]
    out["age_group"] = out["age"].apply(age_to_group)
    return out


def splits_disjoint(*dfs, col="filepath"):
    """True iff no filepath appears in more than one split."""
    seen = set()
    for df in dfs:
        values = set(df[col])
        if seen & values:
            return False
        seen |= values
    return True


def one_off_correct(y_true, y_pred):
    """Fraction of predictions within one age group of the truth."""
    import numpy as np
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float((np.abs(y_true - y_pred) <= 1).mean())


def accuracy_parity_gap(acc_dict):
    """Best-minus-worst group accuracy — our named fairness metric."""
    return max(acc_dict.values()) - min(acc_dict.values())


def fairness_check(acc_dict, max_gap):
    """Weakness detection: True iff the accuracy parity gap is within threshold."""
    return accuracy_parity_gap(acc_dict) <= max_gap


# ---------- deployment decision logic (age-restricted sales) ----------

def p_at_least(probs, band_frac):
    """P(age >= threshold) from 9-band probabilities.

    All probability mass in bands 20-29 and above counts fully. The 10-19 band
    straddles the legal threshold of 18, so only `band_frac` of it counts —
    that fraction is derived from the data, not assumed uniform.
    """
    probs = list(probs)
    return float(sum(probs[3:]) + probs[2] * band_frac)


def decide(p_over, p_approve=0.95, p_reject=0.05):
    """Three-way deployment policy for an age-restricted sale.

    Returns 'auto-clear', 'auto-reject', or 'human-review'. The abstention band
    between the thresholds is what keeps a human in the loop (EU AI Act Art. 14).
    """
    if p_over >= p_approve:
        return "auto-clear"
    if p_over <= p_reject:
        return "auto-reject"
    return "human-review"