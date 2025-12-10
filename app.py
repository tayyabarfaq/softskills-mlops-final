# app.py
import os
from flask import Flask, render_template, request, jsonify
import torch
from transformers import AutoTokenizer, AutoModel
import torch.nn as nn

# ---------------------------
# Config
# ---------------------------
MODEL_NAME = "microsoft/deberta-v3-small"
TOKENIZER_DIR = "./softskills_model"       # folder with tokenizer files
MODEL_WEIGHTS = "softskills_model.pt"      # your saved weights filename
MAX_LENGTH = 256

# ---------------------------
# Model definition
# ---------------------------
class SoftSkillModel(nn.Module):
    def __init__(self, model_name, num_labels=4):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.regressor = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0, :]   # CLS token
        return self.regressor(pooled)

# ---------------------------
# Initialize app and load model+tokenizer
# ---------------------------
app = Flask(__name__, static_folder='static', template_folder='templates')

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR)

# load model
model = SoftSkillModel(MODEL_NAME)
state = torch.load(MODEL_WEIGHTS, map_location="cpu")   # load state dict on CPU first
model.load_state_dict(state)
model.to(device)
model.eval()

# ---------------------------
# Prediction helper
# ---------------------------
def predict_softskills(question: str, answer: str):
    text = f"Q: {question}\nA: {answer}"

    encoding = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH
    )

    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    # outputs shape: (1, 4) -> squeeze -> [p1, p2, p3, p4]
    preds = outputs.squeeze().cpu().tolist()

    # Clip predictions into 1..5 range (your model may not require this step)
    preds = [max(1, min(5, float(p))) for p in preds]

    return {
        "communication": round(preds[0], 2),
        "confidence": round(preds[1], 2),
        "teamwork": round(preds[2], 2),
        "problem_solving": round(preds[3], 2)
    }

# ---------------------------
# Routes
# ---------------------------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    # Accept both JSON and form submissions
    if request.is_json:
        body = request.get_json()
        question = body.get("question", "")
        answer = body.get("answer", "")
    else:
        question = request.form.get("question", "")
        answer = request.form.get("answer", "")

    if not question.strip() or not answer.strip():
        return jsonify({"error": "Both question and answer are required."}), 400

    try:
        result = predict_softskills(question, answer)
        return jsonify({"success": True, "result": result})
    except Exception as e:
        # Keep error message minimal in production
        return jsonify({"success": False, "error": str(e)}), 500

# ---------------------------
# Run
# ---------------------------
if __name__ == "__main__":
    # debug=False for production
    app.run(host="0.0.0.0", port=5000, debug=True)
