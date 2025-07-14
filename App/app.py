from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import joblib

app = Flask(__name__)

# Load saved model and scaler
model = joblib.load('model/model.pkl')
scaler = joblib.load('model/scaler.pkl')

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html', prediction=None)

@app.route('/predict', methods=['POST'])
def predict():
    # Get input values from form
    Income = float(request.form['Income'])
    Kidhome = int(request.form['Kidhome'])
    Teenhome = int(request.form['Teenhome'])
    Recency = int(request.form['Recency'])
    MntWines = float(request.form['MntWines'])
    NumWebPurchases = int(request.form['NumWebPurchases'])

    # Feature engineering: TotalChildren = Kidhome + Teenhome
    TotalChildren = Kidhome + Teenhome

    # Create input DataFrame
    input_data = pd.DataFrame([[
        Income, Kidhome, Recency, MntWines, NumWebPurchases, TotalChildren
    ]], columns=['Income', 'Kidhome', 'Recency', 'MntWines', 'NumWebPurchases', 'TotalChildren'])

    # Scale input
    input_scaled = scaler.transform(input_data)

    # Make prediction
    prediction = model.predict(input_scaled)[0]

    # Pass input back to template
    inputs = {
        'Income': Income,
        'Kidhome': Kidhome,
        'Teenhome': Teenhome,
        'Recency': Recency,
        'MntWines': MntWines,
        'NumWebPurchases': NumWebPurchases
    }

    return render_template('index.html', prediction=prediction, inputs=inputs)

if __name__ == "__main__":
    app.run(debug=True)
