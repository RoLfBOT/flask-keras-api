import numpy as np 
from keras.models import load_model
from flask import Flask, jsonify, render_template, request
from PIL import Image

def loadModel():
    global model
    model = load_model('/predictor.h5')

app = Flask(__name__)

@app.route('/api/mnist', methods = ['POST'])
def mnist():
    img = Image.open(request.files['image'])
    print(img.shape)    
    return jsonify(result="success")

if __name__ == '__main__':
    loadModel()
    app.run()