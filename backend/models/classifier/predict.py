import pickle
import numpy as np

vectorizer    = pickle.load(open("vectorizer.pkl",    "rb"))
stage1        = pickle.load(open("stage1_model.pkl",  "rb"))
stage2_models = pickle.load(open("stage2_models.pkl", "rb"))
thresholds    = pickle.load(open("thresholds.pkl",    "rb"))

HIGH_THRESHOLD = thresholds["high_threshold"]
HIGH_IDX       = thresholds["high_idx"]

# ── Keyword safety override ───────────────────────────────────────────────────
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
}

def keyword_signal(text):
    text = text.lower()
    matches, score = [], 0.0
    for phrase, weight in CRISIS_KEYWORDS.items():
        if phrase in text:
            matches.append(phrase)
            score = max(score, weight)
    return score, matches


# ── Routing ───────────────────────────────────────────────────────────────────
def route_level(score):
    if score >= 8.0:
        return "IMMEDIATE_ESCALATION"
    elif score >= 4.0:
        return "MEDIUM_PRIORITY"
    return "LOW_PRIORITY"


# ── ML explainability (works with LogisticRegression) ─────────────────────────
def explain_stage1(vec, pred_band, top_k=5):
    """Return the top TF-IDF features driving the Stage-1 band prediction."""
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


# ── Score refinement via Stage 2 ─────────────────────────────────────────────
def refine_score(vec, band):
    """
    Given a Stage-1 band, run the band-specific Stage-2 model and return a
    probability-weighted expected score on the 1–10 scale.
    """
    entry = stage2_models.get(band)
    band_midpoints = {"LOW": 2.0, "MEDIUM": 5.5, "HIGH": 9.0}

    if entry is None:
        return band_midpoints[band], None

    kind, payload = entry

    if kind == "constant":
        return float(payload), None

    # kind == "model"
    probs   = payload.predict_proba(vec)[0]
    classes = payload.classes_.astype(float)
    score   = float(np.dot(probs, classes))     # expected value

    score_probs = {
        int(cls): round(float(p), 3)
        for cls, p in zip(payload.classes_, probs)
    }
    return score, score_probs


# ── Main predict function ─────────────────────────────────────────────────────
def predict(text):
    kw_score, kw_matches = keyword_signal(text)

    # Hard safety override — bypass ML entirely
    if kw_score >= 0.95:
        return {
            "score":      10.0,
            "band":       "HIGH",
            "route":      "IMMEDIATE_ESCALATION",
            "confidence": 1.0,
            "triggered_by":    kw_matches,
            "stage1_explanation": None,
            "stage2_score_probs": None,
            "method":     "keyword_override",
        }

    vec = vectorizer.transform([text.lower()])

    # Stage 1: band classification with calibrated HIGH threshold
    s1_proba = stage1.predict_proba(vec)[0]
    if s1_proba[HIGH_IDX] >= HIGH_THRESHOLD:
        band = "HIGH"
    else:
        band = stage1.classes_[int(np.argmax(s1_proba))]
    s1_conf  = float(s1_proba[HIGH_IDX]) if band == "HIGH" else float(np.max(s1_proba))

    # Explainability (run after band is determined)
    s1_expl = explain_stage1(vec, band)

    # Keyword signal can only raise the score, never lower it
    score = max(raw_score, kw_score * 10.0)
    score = float(np.clip(score, 1.0, 10.0))

    return {
        "score":      round(score, 2),
        "band":       band,
        "route":      route_level(score),
        "confidence": round(s1_conf, 3),
        "triggered_by":       kw_matches if kw_matches else None,
        "stage1_explanation": s1_expl,
        "stage2_score_probs": s2_probs,
        "method":             "ml_pipeline",
    }


# ── Quick smoke-test ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    samples = [
        "I've been feeling a bit down lately, not sure what to do.",
        "I can't stop crying and I don't know why I even bother anymore.",
        "I want to hurt myself, I have a plan and I can't stop thinking about it.",
        "I want to kill someone, I have a gun and I'm ready.",
        "Just stressed about work deadlines this week.",
    ]
    for txt in samples:
        r = predict(txt)
        print(f"\nText:   {txt[:70]}")
        print(f"Score:  {r['score']}/10  Band: {r['band']}  Route: {r['route']}")
        print(f"Method: {r['method']}  Confidence: {r['confidence']}")
        if r["triggered_by"]:
            print(f"Keywords: {r['triggered_by']}")