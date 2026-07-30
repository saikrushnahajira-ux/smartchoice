import os
from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)
MODEL_PATH = os.path.join("model", "house_price_model.pkl")

try:
    model = joblib.load(MODEL_PATH)
except Exception:
    model = None

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.get_json() or {}
    area = float(data.get("area", 1500))
    bedrooms = int(data.get("bedrooms", 3))
    bathrooms = int(data.get("bathrooms", 2))
    floors = int(data.get("floors", 1))
    parking = int(data.get("parking", 1))
    age = int(data.get("age", 5))
    location_tier = int(data.get("location_tier", 2))

    features = np.array([[area, bedrooms, bathrooms, floors, parking, age, location_tier]])
    price = model.predict(features)[0] if model else area * 3400

    return jsonify({
        "status": "success",
        "estimated_price": round(price),
        "formatted_price": f"₹{round(price):,}"
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=3000, debug=True)