import pickle
import numpy as np
from pathlib import Path

_DIR = Path(__file__).resolve().parent

vectorizer    = pickle.load(open(_DIR / "vectorizer.pkl",    "rb"))
stage1        = pickle.load(open(_DIR / "stage1_model.pkl",  "rb"))
stage2_models = pickle.load(open(_DIR / "stage2_models.pkl", "rb"))
thresholds    = pickle.load(open(_DIR / "thresholds.pkl",    "rb"))

HIGH_THRESHOLD = thresholds["high_threshold"]
HIGH_IDX       = thresholds["high_idx"]

CRISIS_KEYWORDS = {
    # Self-harm / suicidal
    "kill myself":          1.0,
    "end my life":          1.0,
    "i want to die":        0.95,
    "don't want to live":   0.95,
    "can't go on":          0.9,
    "no reason to live":    0.9,
    "hurt myself":          0.95,
    "suicidal":             0.9,
    "self harm":            0.9,
    # Violence toward others
    "i am going to hurt someone": 1.0,
    "i want to kill someone":     0.9,
    "going to kill them":         0.9,
    "i will shoot someone":       1.0,
    "i will stab someone":        1.0,
    "i have a gun":               1.0,
    "drug":                 0.6
}

def keyword_signal(text):
    text = text.lower()
    matches, score = [], 0.0
    for phrase, weight in CRISIS_KEYWORDS.items():
        if phrase in text:
            matches.append(phrase)
            score = max(score, weight)
    return score, matches


def route_level(score):
    if score >= 8.0:
        return "IMMEDIATE_ESCALATION"
    elif score >= 4.0:
        return "MEDIUM_PRIORITY"
    return "LOW_PRIORITY"


def explain_stage1(vec, pred_band, top_k=5):
    probs      = stage1.predict_proba(vec)[0]
    classes    = stage1.classes_
    pred_idx   = list(classes).index(pred_band)
    coef_row   = stage1.coef_[pred_idx]
    feat_names = vectorizer.get_feature_names_out()

    active_idx   = vec.nonzero()[1]
    scored_feats = sorted(
        [(feat_names[i], float(coef_row[i])) for i in active_idx],
        key=lambda x: abs(x[1]),
        reverse=True
    )
    return {
        "predicted_band": pred_band,
        "band_probabilities": {
            cls: round(float(p), 3) for cls, p in zip(classes, probs)
        },
        "top_features": scored_feats[:top_k],
    }


def refine_score(vec, band):
    entry = stage2_models.get(band)
    band_midpoints = {"LOW": 2.0, "MEDIUM": 5.5, "HIGH": 9.0}

    if entry is None:
        return band_midpoints[band], None

    kind, payload = entry

    if kind == "constant":
        return float(payload), None

    probs   = payload.predict_proba(vec)[0]
    classes = payload.classes_.astype(float)
    score   = float(np.dot(probs, classes))

    score_probs = {
        int(cls): round(float(p), 3)
        for cls, p in zip(payload.classes_, probs)
    }
    return score, score_probs


def predict(text):
    kw_score, kw_matches = keyword_signal(text)

    if kw_score >= 0.95:
        return {
            "score":              10.0,
            "band":               "HIGH",
            "route":              "IMMEDIATE_ESCALATION",
            "confidence":         1.0,
            "triggered_by":       kw_matches,
            "stage1_explanation": None,
            "stage2_score_probs": None,
            "method":             "keyword_override",
        }

    vec = vectorizer.transform([text.lower()])

    s1_proba = stage1.predict_proba(vec)[0]
    if s1_proba[HIGH_IDX] >= HIGH_THRESHOLD:
        band = "HIGH"
    else:
        band = stage1.classes_[int(np.argmax(s1_proba))]
    s1_conf = float(s1_proba[HIGH_IDX]) if band == "HIGH" else float(np.max(s1_proba))

    raw_score, s2_probs = refine_score(vec, band)

    score = max(raw_score, kw_score * 10.0)
    score = float(np.clip(score, 1.0, 10.0))

    s1_expl = explain_stage1(vec, band)

    return {
        "score":              round(score, 2),
        "band":               band,
        "route":              route_level(score),
        "confidence":         round(s1_conf, 3),
        "triggered_by":       kw_matches if kw_matches else None,
        "stage1_explanation": s1_expl,
        "stage2_score_probs": s2_probs,
        "method":             "ml_pipeline",
    }