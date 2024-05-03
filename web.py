from flask import Flask, render_template, request, jsonify
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import pickle

app = Flask(__name__)

# Load your pre-trained model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("lebretou/code-human-ai")
model = AutoModelForSequenceClassification.from_pretrained("lebretou/code-human-ai")
model.eval()  # Set the model to evaluation mode

# Load label encoder
with open('./label_encoder.pkl', 'rb') as file:
    label_encoder = pickle.load(file)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    code = request.form['code']
    encoding = tokenizer(code, truncation=True, padding=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**encoding)
        predictions = outputs.logits.argmax(-1)
    predicted_label = label_encoder.inverse_transform(predictions.numpy())[0]

    # Map your predicted label to an emoji
    emoji = '🤖' if predicted_label == 'AI-written' else '👤'

    return jsonify(emoji=emoji)

if __name__ == '__main__':
    app.run(debug=True)
