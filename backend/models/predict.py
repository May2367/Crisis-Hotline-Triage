import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification

# 1. Load your newly trained model
model_path = "./model"
model = RobertaForSequenceClassification.from_pretrained(model_path)
tokenizer = RobertaTokenizer.from_pretrained(model_path)

model.eval()  # Put the model in "Evaluation" mode

# Define high-priority keywords (case-insensitive)
HIGH_PRIORITY_KEYWORDS = [
    "blood", "weapon", "danger", "bomb", "help", "dying", "bleeding", "attack", "emergency", "suicide", "kill", "hurt"
]

def get_urgency_score(text):
    # 2. Prepare the text
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    
    # 3. Predict
    with torch.no_grad():
        outputs = model(**inputs)
        # The model outputs a raw "logit"
        score = outputs.logits.item()
    
    # 4. Apply keyword boost
    text_lower = text.lower()
    if any(keyword in text_lower for keyword in HIGH_PRIORITY_KEYWORDS):
        score = (10-score)*9/10 + score  # Boost by 3 points for high-priority keywords
    
    # 5. Clean up the score
    # Since we trained 1-10, we clamp it so it doesn't give weird numbers
    final_score = max(1.0, min(10.0, score))
    return round(final_score, 2)

# 4. Test it!
test_phrases = [
    "I'm just making some tea and relaxing.",
    "I feel a bit stressed about my exams tomorrow.",
    "HELP, I can't breathe and my chest is tight!",
    "Please help me, my leg is bleeding out everywhere",
    "Please HELP me!!! my leg is bleeding out everywhere",
    "HELP HELP HELP HELP IM DYING I CANT BREATHE IM DYING",
    "MY DOG IS SO CUTE!!!",
    "There is someone outside with a weapon, please send help now."
]

print("\n--- Model Triage Results ---")
for phrase in test_phrases:
    result = get_urgency_score(phrase)
    print(f"Score: {result} | Text: {phrase}")