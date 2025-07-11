# 🧩 Analysing Customers Personality for Business Improvement

This project predicts whether a customer is likely to respond to a marketing campaign based on their characteristics, using supervised machine learning.

---

## ✅ Features Used
- **Income**
- **Kidhome**
- **Teenhome**
- **Recency**
- **MntWines**
- **NumWebPurchases**

During preprocessing:
- `TotalChildren` = `Kidhome` + `Teenhome`
- `Teenhome` is dropped
- Features are scaled

---

## ⚙️ ML Algorithms Used
- Logistic Regression
- SVM
- Random Forest
- AdaBoost
- Gradient Boosting

---

## 📂 Project Structure

.
├── Data/
│ └── balanced_customer_personality.csv
│ └── preprocessed_customer_personality.csv
│
├── Training/
│ ├── training_notebook.ipynb
│ ├── preprocess_data.ipynb
│
├── Evaluation/
│ ├── evaluation_and_tuning.ipynb
│ ├── best_model_saving.ipynb
│ ├── tuned_models_comparison.csv
│ ├── best_hyperparameters.csv
│
├── App/
│ ├── app.py
│ ├── model/
│ │ ├── model.pkl
│ │ ├── scaler.pkl
│ ├── templates/
│ │ ├── index.html
│ ├── static/
│ │ ├── css/
│ │ ├── js/
│
├── README.md
├── requirements.txt
├── python_version.txt
├── setup.exe (optional)

---

## 🚀 How to Run

1️⃣ Install dependencies:
```bash
pip install -r requirements.txt
2️⃣ Go to App/ and run:

python app.py
3️⃣ Open http://127.0.0.1:5000 and try it!

