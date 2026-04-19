import sys
from pathlib import Path
from flask import Flask, request, jsonify
import whisper

FRONTEND_DIR   = Path(__file__).resolve().parent
CLASSIFIER_DIR = FRONTEND_DIR.parent / "backend" / "models" / "classifier"

sys.path.insert(0, str(CLASSIFIER_DIR))
from predict import predict

app = Flask(__name__, static_folder="public", static_url_path="")
audio_model = whisper.load_model("base")


@app.route('/')
def index():
    return app.send_static_file('index.html')


@app.route('/process_audio', methods=['POST'])
def process_audio():
    try:
        if 'audio' not in request.files:
            return jsonify({"error": "No audio file provided"}), 400

        audio_path = FRONTEND_DIR / "temp_audio.webm"
        request.files['audio'].save(str(audio_path))

        text       = audio_model.transcribe(str(audio_path))['text'].strip()
        prediction = predict(text)
        audio_path.unlink(missing_ok=True)

        return jsonify({"text": text, **prediction})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/process_text', methods=['POST'])
def process_text():
    try:
        text       = request.get_json()['text'].strip()
        prediction = predict(text)
        return jsonify({"text": text, **prediction})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)