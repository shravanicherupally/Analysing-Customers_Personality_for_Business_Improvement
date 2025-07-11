# ✅ app.py

from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import joblib

app = Flask(__name__)

# ---------------------------
# 1️⃣ Load Scaler and Model
# ---------------------------
model = joblib.load('model/model.pkl')
scaler = joblib.load('model/scaler.pkl')

# ---------------------------
# 2️⃣ Home Route
# ---------------------------
@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html', prediction=None)

# ---------------------------
# 3️⃣ Predict Route
# ---------------------------
@app.route('/predict', methods=['POST'])
def predict():
    # Get input values from form
    Income = float(request.form['Income'])
    Kidhome = int(request.form['Kidhome'])
    Teenhome = int(request.form['Teenhome'])
    Recency = int(request.form['Recency'])
    MntWines = float(request.form['MntWines'])
    NumWebPurchases = int(request.form['NumWebPurchases'])

    # ---------------------------
    # Feature Engineering: Add TotalChildren
    # Drop Teenhome to match training
    # ---------------------------
    TotalChildren = Kidhome + Teenhome

    # Create input DataFrame
    input_data = pd.DataFrame([[
        Income, Kidhome, Recency, MntWines, NumWebPurchases, TotalChildren
    ]], columns=['Income', 'Kidhome', 'Recency', 'MntWines', 'NumWebPurchases', 'TotalChildren'])

    # Scale input
    input_scaled = scaler.transform(input_data)

    # Predict
    prediction = model.predict(input_scaled)[0]

    # Return result
    return render_template('index.html', prediction=prediction)

# ---------------------------
# 4️⃣ Run the App
# ---------------------------
if __name__ == "__main__":
    app.run(debug=True)
