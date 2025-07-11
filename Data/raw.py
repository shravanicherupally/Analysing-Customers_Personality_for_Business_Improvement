import pandas as pd
import numpy as np

# Set seed for reproducibility
np.random.seed(42)

# Number of samples per class
n_per_class = 1500

# --- Generate Class 0 (Did NOT respond) ---
class0 = pd.DataFrame({
    'Income': np.random.normal(40000, 8000, n_per_class).astype(int),  # Avg income 40k
    'Kidhome': np.random.randint(0, 3, n_per_class),  # 0–2 kids
    'Teenhome': np.random.randint(0, 3, n_per_class),  # 0–2 teens
    'Recency': np.random.randint(30, 100, n_per_class),  # Days since last purchase
    'MntWines': np.random.normal(100, 50, n_per_class).astype(int),  # Low spending
    'NumWebPurchases': np.random.randint(0, 5, n_per_class),  # Few web purchases
    'Response': 0
})

# --- Generate Class 1 (Did respond) ---
class1 = pd.DataFrame({
    'Income': np.random.normal(70000, 10000, n_per_class).astype(int),  # Avg income 70k
    'Kidhome': np.random.randint(0, 3, n_per_class),
    'Teenhome': np.random.randint(0, 3, n_per_class),
    'Recency': np.random.randint(0, 30, n_per_class),  # Recent purchases
    'MntWines': np.random.normal(500, 100, n_per_class).astype(int),  # High spending
    'NumWebPurchases': np.random.randint(5, 15, n_per_class),  # Frequent web purchases
    'Response': 1
})

# Combine and shuffle
df = pd.concat([class0, class1]).sample(frac=1, random_state=42).reset_index(drop=True)

# Clip any negative spending
df['MntWines'] = df['MntWines'].clip(lower=0)

# Save to CSV
df.to_csv('balanced_customer_personality.csv', index=False)

print('✅ Balanced dataset created and saved as balanced_customer_personality.csv')
print(df.head())
