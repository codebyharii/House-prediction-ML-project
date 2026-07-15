# Backend API - ye Python model ko chalata hai aur frontend se baat karta hai
# Local pe chalane ke liye terminal mein: python app.py

import os
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import joblib
import pandas as pd

# Flask app banao (static_folder="." matlab isi folder se index.html serve hoga)
app = Flask(__name__, static_folder=".")
CORS(app)  # frontend (browser) ko is API se baat karne ki permission deta hai

# Step 1: trained model ko load karo (jo Colab se download kiya tha)
model = joblib.load("house_model.pkl")

# Model ko jo columns chahiye (training ke waqt jo X mein the) - same order mein
FEATURE_COLUMNS = [
    "longitude", "latitude", "housing_median_age", "total_rooms",
    "total_bedrooms", "population", "households", "median_income",
    "ocean_proximity_<1H OCEAN", "ocean_proximity_INLAND",
    "ocean_proximity_ISLAND", "ocean_proximity_NEAR BAY",
    "ocean_proximity_NEAR OCEAN"
]


# Home page: jab koi website kholega to index.html dikhega
@app.route("/")
def home():
    return send_from_directory(".", "index.html")


# Step 2: ek "route" banao jahan frontend request bhejega
@app.route("/predict", methods=["POST"])
def predict():
    # frontend se aaya hua data (JSON form mein)
    data = request.get_json()

    # data ko ek DataFrame row mein badlo (model isi format ko samajhta hai)
    row = pd.DataFrame([data], columns=FEATURE_COLUMNS)

    # Step 3: model se prediction karwao
    predicted_price = model.predict(row)[0]

    # Step 4: answer ko JSON mein wapas frontend ko bhejo
    return jsonify({"predicted_price": round(float(predicted_price), 2)})


# server chalu karo
if __name__ == "__main__":
    # hosting waali jagah PORT khud deti hai; local pe 5000 use hoga
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
