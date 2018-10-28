import numpy as np 
import tensorflow as tf 
from keras.models import load_model
from flask import Flask, jsonify, render_template, request
from PIL import Image

##Flask webServer
app = Flask(__name__)

@app.route('/api/mnist', methods = ['POST'])
def mnist():
    images = request.files.getlist("images")
    img = np.empty((0, 28, 28))

    for x in images:
        image = np.array(Image.open(x)).astype('float32')
        img = np.append(img, [image], axis=0)
    
    img = img.reshape((img.shape[0], 28, 28, 1))
    img /= 255.0

    with graph.as_default():
        preds = model.predict(img, batch_size=img.shape[0]).tolist()
        res = {}
        for i in range(img.shape[0]):
                res[i+1] = preds[i]

        return jsonify(results=res)

##Load Model from saved file
def loadModel():
    global model,graph
    model = load_model('/predictor.h5')
    graph = tf.get_default_graph()


if __name__ == '__main__':
    loadModel()
    app.run()