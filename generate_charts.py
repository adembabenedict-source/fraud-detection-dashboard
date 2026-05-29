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
n_fraud = int(n_samples * 0.03) # 3% fraud rate

normal_amount = np.random.exponential(50, n_samples - n_fraud)
normal_hour = np.random.randint(8, 20, n_samples - n_fraud)

fraud_amount = np.random.exponential(200, n_fraud)
fraud_hour = np.random.randint(0, 24, n_fraud)

amount = np.concatenate([normal_amount, fraud_amount])
hour = np.concatenate([normal_hour, fraud_hour])
fraud = np.array([0]*(n_samples - n_fraud) + [1]*n_fraud)

df = pd.DataFrame({
    'Amount': amount.round(2),
    'Hour': hour,
    'Fraud': fraud
})

# 3. Train model
X = df[['Amount', 'Hour']]
y = df['Fraud']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

# 4. Print metrics for README
print("=== MODEL RESULTS FOR PORTFOLIO ===")
print(classification_report(y_test, y_pred, target_names=['Normal', 'Fraud']))
print("===================================")

# 5. Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(7,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Normal', 'Fraud'],
            yticklabels=['Normal', 'Fraud'])
plt.title('Confusion Matrix - Fraud Detection', fontsize=14, weight='bold')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.savefig('reports/confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.close()

# 6. ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(7,5))
plt.plot(fpr, tpr, linewidth=2, label=f'AUC = {roc_auc:.3f}')
plt.plot([0,1],[0,1],'k--', label='Random')
plt.title('ROC Curve - Fraud Detection', fontsize=14, weight='bold')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.grid(alpha=0.3)
plt.savefig('reports/roc_curve.png', dpi=300, bbox_inches='tight')
plt.close()

print("✅ DONE: 2 charts saved to /reports folder")
print("Check the 'reports' folder in VS Code")