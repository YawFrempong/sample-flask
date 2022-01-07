from flask import Flask, request, jsonify
from torch_utils import initialize_model, inference_new

app = Flask(__name__)
device, tokenizer, model = initialize_model()


@app.route('/', methods=['POST'])
def predict():
    if request.method == 'POST':
        content = request.json
        results = inference_new(device, tokenizer, model, content['text'])
        return jsonify({'result':results})