# ================================================
# 🚗 Smart Driving Model Trainer (Auto-Fix Version)
# ================================================

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# ------------------------------------------------
# STEP 1: Load the dataset
# ------------------------------------------------
data = pd.read_csv(r"D:\DL\DL Project\Dldataset.csv")

print("✅ Dataset Loaded Successfully!")
print("Shape:", data.shape)
print("Columns:", data.columns.tolist())
print("\n📊 Sample Data:")
print(data.head())

# ------------------------------------------------
# STEP 2: Define Features and Target
# ------------------------------------------------
X = data[['AX', 'AY', 'AZ', 'SPD', 'BRK', 'ACC']]
y = data['COND']

# ✅ Automatically adjust target range if not starting from 0
if y.min() != 0:
    print(f"\n⚙️ Normalizing target labels: shifting from {y.min()}–{y.max()} to 0–{y.max()-1}")
    y = y - y.min()

# ------------------------------------------------
# STEP 3: Data Scaling
# ------------------------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# ------------------------------------------------
# STEP 4: Train Multiple Models
# ------------------------------------------------
results = {}

# Random Forest
rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
results['RandomForest'] = accuracy_score(y_test, rf_pred)

# XGBoost
xgb = XGBClassifier(
    n_estimators=250,
    learning_rate=0.05,
    max_depth=8,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric='mlogloss'
)
xgb.fit(X_train, y_train)
xgb_pred = xgb.predict(X_test)
results['XGBoost'] = accuracy_score(y_test, xgb_pred)

# LightGBM
lgb = LGBMClassifier(
    n_estimators=250,
    learning_rate=0.05,
    max_depth=-1,
    random_state=42
)
lgb.fit(X_train, y_train)
lgb_pred = lgb.predict(X_test)
results['LightGBM'] = accuracy_score(y_test, lgb_pred)

# ------------------------------------------------
# STEP 5: Compare Models
# ------------------------------------------------
print("\n📊 Model Accuracy Comparison:")
for model, acc in results.items():
    print(f"{model}: {acc:.4f}")

best_model_name = max(results, key=results.get)
print(f"\n🏆 Best Model Selected: {best_model_name}")

# ------------------------------------------------
# STEP 6: Detailed Performance Report
# ------------------------------------------------
if best_model_name == "RandomForest":
    best_model = rf
    y_pred = rf_pred
elif best_model_name == "XGBoost":
    best_model = xgb
    y_pred = xgb_pred
else:
    best_model = lgb
    y_pred = lgb_pred

print("\n📄 Classification Report:")
print(classification_report(y_test, y_pred))

# ------------------------------------------------
# STEP 7: Feature Importance Visualization
# ------------------------------------------------
importances = best_model.feature_importances_
feature_names = ['AX', 'AY', 'AZ', 'SPD', 'BRK', 'ACC']

plt.figure(figsize=(8, 5))
plt.barh(feature_names, importances, color='teal')
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.title(f"Feature Importance ({best_model_name})")
plt.tight_layout()
plt.show()

# ------------------------------------------------
# STEP 8: Save Model and Scaler
# ------------------------------------------------
joblib.dump(best_model, "driving_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print(f"\n✅ Model saved as driving_model.pkl")
print(f"✅ Scaler saved as scaler.pkl")
print(f"✅ You can now use this model in feedback_engine.py for real-time predictions.")