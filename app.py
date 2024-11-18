from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

# Load the trained model
model = joblib.load("model.pkl")

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    # Get user input
    data = request.form
    features = np.array([[int(data["GRE"]), int(data["TOEFL"]), int(data["University Rating"]),
                          float(data["SOP"]), float(data["LOR"]), float(data["CGPA"]), int(data["Research"])]])
    
    # Predict admission chances
    prediction = model.predict(features)[0]
    
    # Convert to percentage and round to 2 decimal places
    prediction_percentage = round(prediction*10, 2)
    
    return render_template("results.html", prediction=prediction_percentage)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

