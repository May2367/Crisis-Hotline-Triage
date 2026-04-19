import whisper


class SpeechToTextService:
    def __init__(self, model_size="base"):
        self.model = whisper.load_model(model_size)

    def transcribe_file(self, file_path: str) -> str:
        result = self.model.transcribe(file_path)
        return result["text"].strip()