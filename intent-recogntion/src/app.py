#coding for the main application which creates the flask web application


from flask import Flask, render_template, url_for, request
import pickle
from predict import *


# load the model from disk
app = Flask(__name__)


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        message = request.form["message"]
        data = [message]
        my_prediction = predict_class(data)
    return render_template("result.html", prediction=my_prediction)


if __name__ == "__main__":

    app.debug = True
    app.run(host="0.0.0.0", port=5000)
