from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import joblib

app = Flask(__name__)

# Load your saved model and scaler
model = joblib.load('model/model.pkl')
scaler = joblib.load('model/scaler.pkl')

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html', prediction=None)

@app.route('/predict', methods=['POST'])
def predict():
    # ✅ 1. Get form inputs
    Income = float(request.form['Income'])
    Kidhome = int(request.form['Kidhome'])
    Teenhome = int(request.form['Teenhome'])
    Recency = int(request.form['Recency'])
    MntWines = float(request.form['MntWines'])
    NumWebPurchases = int(request.form['NumWebPurchases'])

    # ✅ 2. Feature engineering — total children
    TotalChildren = Kidhome + Teenhome

    # ✅ 3. Prepare data in correct order & shape
    input_df = pd.DataFrame([[
        Income, Kidhome, Recency, MntWines, NumWebPurchases, TotalChildren
    ]], columns=['Income', 'Kidhome', 'Recency', 'MntWines', 'NumWebPurchases', 'TotalChildren'])

    # ✅ 4. Scale
    input_scaled = scaler.transform(input_df)

    # ✅ 5. Predict
    prediction = model.predict(input_scaled)[0]

    # ✅ 6. Package original inputs for display
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
