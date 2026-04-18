import pandas as pd
import pickle
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

BASE_DIR = Path(__file__).resolve().parents[3]
csv_path = BASE_DIR / "datasets" / "text_samples" / "sample_text.csv"

df = pd.read_csv(csv_path)

def map_label(value):
    if 1 <= value <= 3:
        return "LOW"
    elif 4 <= value <= 7:
        return "MEDIUM"
    elif 8 <= value <= 10:
        return "HIGH"
    else:
        raise ValueError(f"Unexpected label: {value}")

df["label"] = df["label"].apply(map_label)

print("\nLabel distribution:")
print(df["label"].value_counts())

X = df["text"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    C=1.0
)

model.fit(X_train_vec, y_train)

preds = model.predict(X_test_vec)

print("\nClassification Report:")
print(classification_report(y_test, preds))

probs = model.predict_proba(X_test_vec[:3])
print("\nSample probability check (first 3 test samples):")
print(probs)

pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))
