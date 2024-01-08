import numpy as np
import pickle
from flask import Flask, request, jsonify, render_template

app = Flask(__name__, static_folder='static')

model = pickle.load(open('modelmay.pkl', 'rb'))

# @app.route("/")
# def Home():
#     return render_template('model.html')

# @app.route("/Charts")
# def Charts():
#     return render_template('linechart.html')

@app.route("/")
def form():
    return render_template('model.html')

@app.route('/', methods=["POST"])
def predict():
    float_feature = [float(x) for x in request.form.values()]
    features = [np.array(float_feature)]
    prediction = model.predict(features)
    return render_template("model.html", prediction_svm = "Hasil Prediksi SVM {}".format(prediction[0]))

if __name__ =="__main__":
    app.run(debug=True)