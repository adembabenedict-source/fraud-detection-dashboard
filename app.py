import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report

# 1. Create reports folder
os.makedirs('reports', exist_ok=True)

# 2. Generate realistic sample fraud data
np.random.seed(42)
n_samples = 5000
n_fraud = int(n_samples * 0.03) # 3% fraud rate = imbalanced

# Normal transactions
normal_amount = np.random.exponential(50, n_samples - n_fraud)
normal_hour = np.random.randint(8, 20, n_samples - n_fraud) # business hours

# Fraud transactions
fraud_amount = np.random.exponential(200, n_fraud) # higher amounts
fraud_hour = np.random.randint(0, 24, n_fraud) # any time

# Combine
amount = np.concatenate([normal_amount, fraud_amount])
hour = np.concatenate([normal_hour, fraud_hour])
fraud = np.array([0]*(n_samples - n_fraud) + [1]*n_fraud)

df = pd.DataFrame({
    'Amount': amount.round(2),
    'Hour': hour,
    'Fraud': fraud
})

# 3. Train simple model for demo
X = df[['Amount', 'Hour']]
y = df['Fraud']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced