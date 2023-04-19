import random
import json

import torch
from flask import Flask, render_template, request

from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)

FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "Sam"
questions = []

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/get')
def get_bot_response():
    global questions
    userText = request.args.get('msg')
    questions.append(userText)
    response = get_response(userText)
    return str(response)

@app.route('/predict', methods=['POST'])
def predict():
    userText = request.json['message']
    response = get_response(userText)
    return {'answer': response}

@app.route('/stats')
def get_stats():
    global questions
    question_counts = {}
    for question in questions:
        if question not in question_counts:
            question_counts[question] = 1
        else:
            question_counts[question] += 1
    sorted_questions = sorted(question_counts.items(), key=lambda x: x[1], reverse=True)
    return render_template('stats.html', questions=sorted_questions)

def get_response(msg):
    sentence = tokenize(msg)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                return random.choice(intent['responses'])
    
    return "No entiendo :("

if __name__ == "__main__":
    app.run(debug=True)