# Lifeline Triage AI

Crisis hotlines regularly face a mismatch between call volume and available staff, meaning the most at-risk callers may not receive help quickly enough. Lifeline Triage AI is a multimodal AI triage assistant that scores incoming calls or messages on a 1–10 urgency scale in real time, so operators can prioritise callers who need immediate intervention first.

> **Status:** prototype and demo — not yet deployed in a live crisis line environment.

---

## How It Works

1. **Input** — the user speaks into a microphone or types directly into the web interface
2. **Transcription** — spoken audio is transcribed to text using OpenAI Whisper
3. **Scoring** — the text is passed through a trained ML classifier that outputs a urgency score from 1 to 10
4. **Output** — the interface displays the score alongside a band label, confidence, key contributing terms, and urgency level

---

## Urgency Scale

The model outputs a continuous score from 1 to 10. This is grouped into three bands for operator routing:

| Score | Band | Description |
|-------|------|-------------|
| 1–3 | LOW | Baseline or mild distress — routine monitoring |
| 4–7 | MEDIUM | Moderate to significant distress — elevated attention |
| 8–10 | HIGH | Severe crisis, active ideation, or attempt in progress — immediate response |

The classifier is optimised to minimise false negatives on HIGH-band cases. Missing a genuine crisis is treated as a more serious error than a false alarm.

---

## ML Architecture

The classifier uses a two-stage pipeline:

**Stage 1 — band classifier**
A TF-IDF vectoriser (unigrams + bigrams) feeds a logistic regression model that routes each input to LOW, MEDIUM, or HIGH. A calibrated probability threshold is selected at training time to keep the HIGH false-negative rate below a configurable ceiling (`MAX_ACCEPTABLE_FNR`, default 0.05).

**Stage 2 — per-band refiners**
A separate logistic regression model is trained for each band and produces a fine-grained score within that band's range (1–3, 4–7, or 8–10). The final score is a probability-weighted expected value across the band's label classes.

**Current limitations**
- Uses TF-IDF rather than transformer-based sentence embeddings — no GPU required, runs on standard hardware including Chrome OS
- Training data is synthetically generated due to difficulty sourcing labelled real-world crisis call data, which limits coverage of edge cases and informal language
- A separate branch (`pytorchBranch_Vinay`) contains an experimental version using a stronger embedding model trained on a smaller dataset — not yet merged

---

## Project Structure

```
lifeline-triage-ai/
├── frontend/
│   ├── public/
│   │   └── index.html
│   └── app.py               # Flask server — audio + text endpoints, Whisper transcription
├── backend/
│   ├── models/
│   │   └── classifier/
│   │       ├── train.py         # Model training pipeline
│   │       └── predict.py       # Inference interface
│   └── text_to_speech.py
├── datasets/
│   └── text_samples/
│       └── sample_text.csv      # Labeled training data (scores 1–10)
├── requirements.txt
└── README.md
```

> Model artifacts (`vectorizer.pkl`, `stage1_model.pkl`, `stage2_models.pkl`, `thresholds.pkl`) are generated locally when you run `train.py` and are not committed to the repository.

---

## Setup

**Install dependencies**

```bash
pip install -r requirements.txt
```

**PyTorch (CPU-only — works on Chrome OS and most laptops)**

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

---

## Running the Project

**Step 1 — train the classifier**

```bash
cd backend/models/classifier
python train.py
```

This trains both pipeline stages, runs threshold calibration, prints evaluation metrics, and saves the model artifacts to the same directory.

**Step 2 — start the frontend**

```bash
cd frontend
python app.py
```

Then open your browser and navigate to the address shown in the terminal output.

---

## Training Notes

The `MAX_ACCEPTABLE_FNR` constant at the top of `train.py` controls the safety/precision trade-off:

```python
MAX_ACCEPTABLE_FNR = 0.05  # max tolerable HIGH false-negative rate
```

Lowering this value routes more inputs to HIGH (better recall, more false alarms). Raising it reduces false alarms at the cost of potentially missing genuine crises. For a production crisis context, keeping this value low is strongly recommended.

---

## Data

Training data is at `datasets/text_samples/sample_text.csv`. Each row contains a short text sample and a label from 1 to 10. The dataset covers:

- Everyday and low-distress language (1–3)
- Moderate stress, anxiety, and emotional difficulty (4–7)
- Severe crisis, suicidal ideation, active attempts, and trauma disclosure (8–10)

The data was synthetically generated. Contributions of real, consented, and appropriately anonymised crisis language data would significantly improve model robustness.

---

## Contributing

Pull requests are welcome. If you are working with crisis line data or have domain expertise in mental health triage, we are particularly interested in collaboration around dataset quality and evaluation methodology.

Please be mindful that this repository contains synthetic examples of crisis and trauma language. Handle all data and model outputs with appropriate care.

---

## Acknowledgements

Built with [scikit-learn](https://scikit-learn.org), [OpenAI Whisper](https://github.com/openai/whisper), and [Flask](https://flask.palletsprojects.com).
