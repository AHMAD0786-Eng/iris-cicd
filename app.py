from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)
model = joblib.load("model.pkl")
names = ["setosa", "versicolor", "virginica"]

@app.route("/")
def home():
    return "Iris ML API - version 1"

@app.route("/predict")
def predict():
    text = request.args.get("values", "5.1,3.5,1.4,0.2")
    values = [float(v) for v in text.split(",")]
    result = model.predict([values])[0]
    return jsonify({"input": values, "prediction": names[result]})
