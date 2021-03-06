from flask import Flask, jsonify, request
from prediction import getPrediction

# initialise the flask
app = Flask(__name__)

# create root
@app.route('/predict-digit', methods = ['POST'])
def predictData():
    image = request.files.get('digit')
    prediction = getPrediction(image)
    return jsonify({'prediction': prediction}), 200

if __name__ == '__main__':
    app.run(debug = True)