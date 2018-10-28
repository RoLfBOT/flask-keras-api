import numpy as np 
import tensorflow as tf 
from keras.models import load_model
from flask import Flask, jsonify, render_template, request
from PIL import Image

def loadModel():
    global model,graph
    model = load_model('/predictor.h5')
    graph = tf.get_default_graph()

app = Flask(__name__)

@app.route('/api/mnist', methods = ['POST'])
def mnist():
    img = np.array(Image.open(request.files['image']))
    img = img.reshape((28, 28, 1)).astype('float32')
    img /=255.0
    with graph.as_default():
        return jsonify(results=model.predict(np.expand_dims(img, axis=0)).tolist())


if __name__ == '__main__':
    loadModel()
    app.run()