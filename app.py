# Credit for help with Model Deployment: https://clarusway.com/model-deployment-with-flask-part-1/ 
# Import Libraries
import numpy as np
from flask import Flask, request, render_template
import pickle

# create Flask app

app= Flask(__name__)

# load Pickle model

model = pickle.load(open("model.pkl", "rb"))

# define Home page

@app.route("/")
def Home():
    return render_template("indexx.html")

# prediction page

@app.route("/predict", methods=["POST"])
def predict():
    float_features = [float(x) for x in request.form.values()]
    features = [np.array(float_features)]
    prediction = model.predict(features)

    return render_template("indexx.html", prediction_text="The flower species is {}".format(prediction))

if __name__ == "__main__":
    app.run(debug=True)