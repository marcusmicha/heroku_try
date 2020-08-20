from flask import Flask, request
from joblib import load
import sklearn
import numpy as np
import json

PATH ='/logregmnist.joblib'

lr_loaded = load(PATH)

app = Flask(__name__)

@app.route('/logreg/predict/')
def predict():
    digit_raw = str(request.args.get('input'))
    # Decoding the string received to a numpy array in order to get the prediction
    digit = digit_raw.split(",")
    prediction = lr_loaded.predict(sklearn.preprocessing.normalize([np.array(digit)]))[0]
    return prediction

@app.route('/predict/', methods=['POST'])
def predict2():
    digits_raw = request.get_json()
    digits = json.loads(digits_raw)
    # List of digits as dict
    digits = digits['inputs']
    digits = [list(dig.items())[0][1] for dig in digits]
    digits = [digit.split(",") for digit in digits]
    digits = np.array(digits)
    predictions = lr_loaded.predict(sklearn.preprocessing.normalize(digits))
    rand_sample_string = ','.join(predictions)
    return rand_sample_string


if __name__ == '__main__':
    app.run()
