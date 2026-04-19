import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, mean_absolute_error, mean_squared_error, r2_score,
    confusion_matrix, classification_report, make_scorer, recall_score
)

BASE_DIR = Path(__file__).resolve().parents[3]
OUT_DIR  = Path(__file__).resolve().parent

csv_path = BASE_DIR / "datasets" / "text_samples" / "sample_text.csv"
df = pd.read_csv(csv_path)
df["text"]  = df["text"].str.lower()
df["label"] = df["label"].astype(float)

# ── Tuneable safety knob ──────────────────────────────────────────────────────
# Lower = more aggressive HIGH routing (higher recall, higher FPR)
# Raise to reduce false alarms once FNR is acceptably low
MAX_ACCEPTABLE_FNR = 0.05

# ── Band assignment ───────────────────────────────────────────────────────────
def assign_band(score):
    if score <= 3:
        return "LOW"
    elif score <= 7:
        return "MEDIUM"
    return "HIGH"

df["band"] = df["label"].apply(assign_band)

print("\nLabel distribution (1–10):")
print(df["label"].value_counts().sort_index())
print("\nBand distribution:")
print(df["band"].value_counts())

# ── Train/test split ──────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df[["label", "band"]],
    test_size=0.2,
    random_state=42,
    stratify=df["band"]
)
y_train_label = y_train["label"]
y_test_label  = y_test["label"]
y_train_band  = y_train["band"]
y_test_band   = y_test["band"]

# ── Shared TF-IDF vectorizer ──────────────────────────────────────────────────
vectorizer  = TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_df=0.95, sublinear_tf=True)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec  = vectorizer.transform(X_test)

# ── Stage 1: coarse band classifier ──────────────────────────────────────────
stage1 = LogisticRegression(max_iter=2000, C=5.0)
stage1.fit(X_train_vec, y_train_band)

print(f"\n── Stage 1 cross-validation (5-fold, stratified) ──")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for metric, scoring in [("accuracy", "accuracy"), ("macro recall", "recall_macro"), ("macro F1", "f1_macro")]:
    scores = cross_val_score(stage1, X_train_vec, y_train_band, cv=cv, scoring=scoring)
    print(f"  {metric:>14}: {scores.mean():.4f} ± {scores.std():.4f}")

def high_recall(y_true, y_pred):
    classes = ["LOW", "MEDIUM", "HIGH"]
    scores  = recall_score(y_true, y_pred, labels=classes, average=None, zero_division=0)
    return scores[2]

high_cv = cross_val_score(stage1, X_train_vec, y_train_band, cv=cv, scoring=make_scorer(high_recall))
print(f"  {'HIGH recall':>14}: {high_cv.mean():.4f} ± {high_cv.std():.4f}  ← most important")

# ── Threshold calibration ─────────────────────────────────────────────────────
HIGH_IDX = list(stage1.classes_).index("HIGH")
s1_proba = stage1.predict_proba(X_test_vec)

print(f"\n── Threshold sweep for HIGH (test set) ──")
print(f"  {'Threshold':>10}  {'HIGH FNR':>10}  {'HIGH FPR':>10}  {'Accuracy':>10}  {'Selected':>10}")

bands = ["LOW", "MEDIUM", "HIGH"]
best_thresh = 0.5
best_fpr    = 1.0
sweep_results = []

for thresh in np.arange(0.20, 0.55, 0.05):
    preds = []
    for probs in s1_proba:
        if probs[HIGH_IDX] >= thresh:
            preds.append("HIGH")
        else:
            preds.append(stage1.classes_[np.argmax(probs)])
    preds = np.array(preds)

    cm_t = confusion_matrix(y_test_band, preds, labels=bands)
    tp   = cm_t[2, 2]
    fn   = cm_t[2, :].sum() - tp
    fp   = cm_t[:, 2].sum() - tp
    tn   = cm_t.sum() - tp - fn - fp
    fnr  = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    fpr  = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    acc  = accuracy_score(y_test_band, preds)
    sweep_results.append((thresh, fnr, fpr, acc, preds))

    # Select: lowest FPR among thresholds where FNR <= MAX_ACCEPTABLE_FNR
    if fnr <= MAX_ACCEPTABLE_FNR and fpr < best_fpr:
        best_fpr    = fpr
        best_thresh = thresh

for thresh, fnr, fpr, acc, _ in sweep_results:
    selected = " ← selected" if thresh == best_thresh else ""
    print(f"  {thresh:>10.2f}  {fnr:>10.4f}  {fpr:>10.4f}  {acc:>10.4f}  {selected}")

print(f"\n  → Selected threshold: {best_thresh:.2f} "
      f"(lowest FPR with FNR ≤ {MAX_ACCEPTABLE_FNR})")

def apply_threshold(proba, threshold, high_idx, classes):
    preds = []
    for probs in proba:
        if probs[high_idx] >= threshold:
            preds.append("HIGH")
        else:
            preds.append(classes[np.argmax(probs)])
    return np.array(preds)

s1_preds = apply_threshold(s1_proba, best_thresh, HIGH_IDX, stage1.classes_)

print(f"\n── Stage 1 (band classifier, threshold={best_thresh:.2f}) ──")
print(f"Accuracy: {accuracy_score(y_test_band, s1_preds):.4f}")

print("\nPer-band FNR / FPR (Stage 1):")
cm = confusion_matrix(y_test_band, s1_preds, labels=bands)
for i, band in enumerate(bands):
    tp  = cm[i, i]
    fn  = cm[i, :].sum() - tp
    fp  = cm[:, i].sum() - tp
    tn  = cm.sum() - tp - fn - fp
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    print(f"  {band:6s}  FNR={fnr:.4f} (missed real {band})   FPR={fpr:.4f} (false alarm as {band})")

print("\nFull classification report (Stage 1):")
print(classification_report(y_test_band, s1_preds, labels=bands, digits=4))

# ── Stage 2: one fine-grained classifier per band ────────────────────────────
BAND_RANGES   = {"LOW": (1, 3), "MEDIUM": (4, 7), "HIGH": (8, 10)}
stage2_models = {}

print(f"── Stage 2 (per-band refiners) ──")
for band, (lo, hi) in BAND_RANGES.items():
    mask_train = y_train_band == band
    mask_test  = y_test_band  == band

    Xb_train = X_train_vec[mask_train]
    yb_train = y_train_label[mask_train]
    Xb_test  = X_test_vec[mask_test]
    yb_test  = y_test_label[mask_test]

    if Xb_train.shape[0] == 0:
        print(f"  {band}: no training samples — skipping")
        continue

    if len(yb_train.unique()) < 2:
        majority = int(yb_train.mode()[0])
        stage2_models[band] = ("constant", majority)
        print(f"  {band}: only 1 class ({majority}) — constant predictor")
        continue

    m = LogisticRegression(max_iter=2000, C=5.0)
    m.fit(Xb_train, yb_train)
    stage2_models[band] = ("model", m)

    if Xb_test.shape[0] > 0:
        preds = m.predict(Xb_test)
        n     = len(preds)
        under = int(np.sum(preds < yb_test))
        over  = int(np.sum(preds > yb_test))
        print(f"  {band}: accuracy={accuracy_score(yb_test, preds):.4f}  "
              f"MAE={mean_absolute_error(yb_test, preds):.4f}  "
              f"under={under}/{n} ({under/n:.2%})  over={over}/{n} ({over/n:.2%})  "
              f"(n_test={n})")
    else:
        print(f"  {band}: no test samples for evaluation")

# ── Full-system evaluation ────────────────────────────────────────────────────
def pipeline_predict_score(vec, band_pred):
    entry = stage2_models.get(band_pred)
    if entry is None:
        return {"LOW": 2.0, "MEDIUM": 5.5, "HIGH": 9.0}[band_pred]
    kind, payload = entry
    if kind == "constant":
        return float(payload)
    probs   = payload.predict_proba(vec)[0]
    classes = payload.classes_
    return float(np.dot(probs, classes.astype(float)))

system_preds = np.array([
    np.clip(pipeline_predict_score(X_test_vec[i], s1_preds[i]), 1, 10)
    for i in range(X_test_vec.shape[0])
])
y_true_arr = np.array(y_test_label)

print(f"\n── Full system (end-to-end regression, 1–10) ──")
print(f"MAE:  {mean_absolute_error(y_true_arr, system_preds):.4f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_true_arr, system_preds)):.4f}")
print(f"R²:   {r2_score(y_true_arr, system_preds):.4f}")

actual_high = (y_true_arr >= 8)
pred_high   = (system_preds >= 8)
tp_e2e = int(np.sum( actual_high &  pred_high))
fn_e2e = int(np.sum( actual_high & ~pred_high))
fp_e2e = int(np.sum(~actual_high &  pred_high))
tn_e2e = int(np.sum(~actual_high & ~pred_high))
fnr_e2e = fn_e2e / (fn_e2e + tp_e2e) if (fn_e2e + tp_e2e) > 0 else 0.0
fpr_e2e = fp_e2e / (fp_e2e + tn_e2e) if (fp_e2e + tn_e2e) > 0 else 0.0
recall  = tp_e2e / (tp_e2e + fn_e2e) if (tp_e2e + fn_e2e) > 0 else 0.0
prec    = tp_e2e / (tp_e2e + fp_e2e) if (tp_e2e + fp_e2e) > 0 else 0.0

print(f"\n── End-to-end safety stats (HIGH = score ≥ 8) ──")
print(f"  TP={tp_e2e}  FN={fn_e2e}  FP={fp_e2e}  TN={tn_e2e}")
print(f"  FNR (missed HIGH):  {fnr_e2e:.4f}  ← minimise this")
print(f"  FPR (false alarms): {fpr_e2e:.4f}")
print(f"  Recall:    {recall:.4f}")
print(f"  Precision: {prec:.4f}")

print("\nSample predictions (pred | actual | band | correct_band):")
for i in range(min(8, len(system_preds))):
    correct_band = assign_band(y_test_label.iloc[i])
    mismatch     = " ← BAND MISMATCH" if s1_preds[i] != correct_band else ""
    print(f"  pred={system_preds[i]:.2f}  actual={y_test_label.iloc[i]:.0f}  "
          f"pred_band={s1_preds[i]}  correct_band={correct_band}{mismatch}")

print("\n── Stage 1 confusion matrix ──")
print(f"{'':>10}", end="")
for b in bands:
    print(f"  pred_{b:6}", end="")
print()
for i, b in enumerate(bands):
    print(f"actual_{b:6}", end="")
    for j in range(len(bands)):
        print(f"  {cm[i,j]:>10}", end="")
    print()

mask_med_test = np.array([assign_band(v) for v in y_test_label]) == "MEDIUM"
med_actual    = y_test_label[mask_med_test].values
med_pred      = system_preds[mask_med_test]
under_mask    = med_pred < med_actual
DELTA_THRESHOLD = 0.5
meaningful = under_mask & ((med_actual - med_pred) >= DELTA_THRESHOLD)

print(f"\n── MEDIUM under-triage detail (delta ≥ {DELTA_THRESHOLD}, predicted < actual) ──")
if meaningful.sum() == 0:
    print("  None — all misses within acceptable margin")
else:
    for a, p in sorted(
        zip(med_actual[meaningful], med_pred[meaningful]),
        key=lambda x: x[0] - x[1], reverse=True
    ):
        print(f"  actual={a:.0f}  pred={p:.2f}  delta={a-p:.2f}")
print(f"  ({int(meaningful.sum())} significant misses out of {int(under_mask.sum())} total under-predictions)")

# ── Persist artifacts ─────────────────────────────────────────────────────────
pickle.dump(vectorizer,    open(OUT_DIR / "vectorizer.pkl",    "wb"))
pickle.dump(stage1,        open(OUT_DIR / "stage1_model.pkl",  "wb"))
pickle.dump(stage2_models, open(OUT_DIR / "stage2_models.pkl", "wb"))
pickle.dump({"high_threshold": best_thresh,
             "high_idx":       HIGH_IDX},
                           open(OUT_DIR / "thresholds.pkl",    "wb"))

print("\nSaved: vectorizer.pkl, stage1_model.pkl, stage2_models.pkl, thresholds.pkl")
print(f"Output directory: {OUT_DIR}")