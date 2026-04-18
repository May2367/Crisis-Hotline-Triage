import pickle

model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

def predict_with_confidence(text):
    vec = vectorizer.transform([text])

    probs = model.predict_proba(vec)[0]
    pred = model.predict(vec)[0]

    return {
        "prediction": pred,
        "probabilities": probs.tolist()
    }