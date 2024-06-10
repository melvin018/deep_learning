from flask import Flask, request, jsonify
import numpy as np
from tensorflow.keras.models import load_model
import cv2

app = Flask(__name__)

# Load the COVID-19 detection model
model_cnn_covid = load_model('cnn.h5')

class_map ={0:"Normal",
            1:"Non-Covid",
            2:"Covid"}

@app.route('/predict_covid', methods=['POST','GET'])
def predict_covid():
    size = 64
    data = cv2.imread('upload\covid_sample.png')
    data = cv2.resize(data,(size,size))
    data = data/255
    
    prediction_cnn = model_cnn_covid.predict(np.array([data]))
    covid_class = class_map[np.argmax(prediction_cnn)]
    return jsonify({
        'cnn_prediction': covid_class
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
