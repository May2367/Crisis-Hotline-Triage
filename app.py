from flask import Flask, render_template, request, jsonify
import whisper
import torch
from transformers import RobertaConfig, RobertaForSequenceClassification, PreTrainedTokenizerFast
from safetensors.torch import load_file
from pathlib import Path

app = Flask(__name__)
audio_model = whisper.load_model("base")

base_dir = Path(__file__).resolve().parent
model_path = base_dir / "backend" / "models" / "model"

if not model_path.exists():
    raise FileNotFoundError(f"Model folder not found at {model_path}")

config = RobertaConfig.from_json_file(model_path / "config.json")
tokenizer = PreTrainedTokenizerFast(tokenizer_file=str(model_path / "tokenizer.json"))

state_dict = load_file(str(model_path / "model.safetensors"))
converted_state_dict = {}
for key, value in state_dict.items():
    key = key.replace(".gamma", ".weight").replace(".beta", ".bias")
    converted_state_dict[key] = value

model = RobertaForSequenceClassification(config)
model.load_state_dict(converted_state_dict, strict=False)
model.eval()

def get_urgency_category(score):
    if score < 4: return "Low"
    if score < 7: return "Medium"
    return "High"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_audio', methods=['POST'])
def process_audio():
    print("Received audio request")
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file provided"}), 400
    
    audio_file = request.files['audio']
    audio_path = "temp_audio.wav"
    audio_file.save(audio_path)
    print("Audio saved")

    # 1. Speech to Text
    result = audio_model.transcribe(audio_path)
    text = result['text'].lower()
    print(f"Transcription done: {text}")

    # 2. Get Score from RoBERTa
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
        score = outputs.logits.item()
    print(f"Prediction done: {score}")
    
    # 3. Clean up the score
    final_score = max(1.0, min(10.0, score))
    score = round(final_score, 2)
    category = get_urgency_category(score)

    # Clean up temp file
    import os
    os.remove(audio_path)
    print("Temp file removed")

    return jsonify({"text": text, "score": score, "category": category})

if __name__ == '__main__':
    app.run(debug=True)