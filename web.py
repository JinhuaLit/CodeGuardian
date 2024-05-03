from flask import Flask, render_template, request, jsonify
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import pickle
import numpy as np

app = Flask(__name__)

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("lebretou/code-human-ai")
model = AutoModelForSequenceClassification.from_pretrained("lebretou/code-human-ai")
model.eval()  # Ensure the model is in evaluation mode

# Load label encoder
with open('label_encoder.pkl', 'rb') as file:
    label_encoder = pickle.load(file)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    code = request.form['code']
    encoding = tokenizer(code, truncation=True, padding=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**encoding, output_attentions=True)
        predictions = outputs.logits.argmax(-1)
    predicted_label = label_encoder.inverse_transform(predictions.cpu().numpy())[0]
    emoji = '🤖' if predicted_label == 'AI-written' else '👤'

    # Process attention weights to calculate line weights
    attention_weights = outputs.attentions[-1].squeeze().mean(dim=0).cpu().numpy()
    input_ids = encoding['input_ids'].cpu().numpy()[0]
    lines = code.split('\n')

    line_attention_weights = []
    current_line_weight = 0
    line_idx = 0

    # Compute attention weight per line
    for token_id, weight in zip(input_ids, attention_weights):
        if token_id == tokenizer.eos_token_id:
            break

        token = tokenizer.decode([token_id])
        if token == tokenizer.pad_token:
            continue

        if token == '\n':
            line_attention_weights.append((line_idx, current_line_weight))
            line_idx += 1
            current_line_weight = 0
        else:
            current_line_weight += weight.sum()

    if current_line_weight > 0:
        line_attention_weights.append((line_idx, current_line_weight))

    # Normalize and sort line weights
    total_weight = sum(weight for _, weight in line_attention_weights)
    line_attention_weights = [(idx, weight / total_weight) for idx, weight in line_attention_weights]
    top_lines_indices = sorted(line_attention_weights, key=lambda x: x[1], reverse=True)[:5]

    # Prepare top lines for JSON response (sending indices only)
    top_lines_indices_list = [idx for idx, _ in top_lines_indices]

    return jsonify(emoji=emoji, topLines=top_lines_indices_list, label=predicted_label)

if __name__ == '__main__':
    app.run(debug=True)