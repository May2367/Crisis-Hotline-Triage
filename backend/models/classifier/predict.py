import pickle
import numpy as np

model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

def safety_override(text):
    text = text.lower()

    high_risk_keywords = [
        "kill myself",
        "end my life",
        "i want to die",
        "can't go on",
        "no reason to live",
        "hurt myself",
        "suicidal"
    ]

    for phrase in high_risk_keywords:
        if phrase in text:
            return {
                "class": "HIGH",
                "confidence": 1.0,
                "risk_score": 100,
                "triggered_by": "safety_override"
            }

    return None

def compute_risk_score(probs, classes):
    class_to_weight = {
        "LOW": 20,
        "MEDIUM": 55,
        "HIGH": 90
    }

    score = 0

    for cls, prob in zip(classes, probs):
        score += class_to_weight[cls] * prob

    return round(score, 2)

def predict(text):

    override = safety_override(text)
    if override:
        return override

    vec = vectorizer.transform([text])

    probs = model.predict_proba(vec)[0]
    classes = model.classes_

    pred_class = classes[np.argmax(probs)]
    confidence = float(np.max(probs))

    risk_score = compute_risk_score(probs, classes)

    return {
        "class": pred_class,
        "confidence": round(confidence, 3),
        "risk_score": risk_score,
        "probabilities": {
            cls: round(float(prob), 3)
            for cls, prob in zip(classes, probs)
        }
    }