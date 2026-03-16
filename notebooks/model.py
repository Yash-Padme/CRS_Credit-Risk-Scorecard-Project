# ═══════════════════════════════════════════════
# CREDIT RISK SCORECARD - MODEL TRAINING
# Complete pipeline: Load → Clean → Train → Save
# ═══════════════════════════════════════════════

import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report, roc_curve
from imblearn.over_sampling import SMOTE

print("=" * 70)
print("CREDIT RISK SCORECARD - MODEL TRAINING PIPELINE")
print("=" * 70)


# ═══════════════════════════════════════════════
# STEP 1: LOAD DATA
# ═══════════════════════════════════════════════

print("\n✅ STEP 1: LOADING DATA...")

try:
    df = pd.read_csv('../data/cs-training.csv', index_col=0)
    print(f"   ✓ Dataset shape: {df.shape}")
    print(f"   ✓ Default rate: {df['SeriousDlqin2yrs'].mean()*100:.2f}%")
except FileNotFoundError:
    print("   ❌ Error: cs-training.csv not found in ../data/ folder")
    print("   Please download from: https://www.kaggle.com/datasets/brycecf/give-me-some-credit")
    exit()


# ═══════════════════════════════════════════════
# STEP 2: DATA CLEANING
# ═══════════════════════════════════════════════

print("\n✅ STEP 2: CLEANING DATA...")

# Make a copy to avoid warnings
df = df.copy()

# Remove impossible age values
df = df[df['age'].between(18, 100)]

# Cap outliers at 99th percentile
p99_rev = df['RevolvingUtilizationOfUnsecuredLines'].quantile(0.99)
df['RevolvingUtilizationOfUnsecuredLines'] = df['RevolvingUtilizationOfUnsecuredLines'].clip(0, p99_rev)

p99_debt = df['DebtRatio'].quantile(0.99)
df['DebtRatio'] = df['DebtRatio'].clip(0, p99_debt)

p99_inc = df['MonthlyIncome'].quantile(0.99)
df['MonthlyIncome'] = df['MonthlyIncome'].clip(0, p99_inc)

# Cap late payment counts at 10
for col in ['NumberOfTime30-59DaysPastDueNotWorse',
            'NumberOfTimes90DaysLate',
            'NumberOfTime60-89DaysPastDueNotWorse']:
    df[col] = df[col].clip(0, 10)

# Handle missing monthly income by age group
df['age_group'] = pd.cut(df['age'],
    bins=[18,35,50,65,100],
    labels=['Young','Middle','Senior','Elder'])

income_medians = df.groupby('age_group', observed=True)['MonthlyIncome'].median()

def fill_income(row):
    if pd.isnull(row['MonthlyIncome']):
        return income_medians[row['age_group']]
    return row['MonthlyIncome']

df['MonthlyIncome'] = df.apply(fill_income, axis=1)

# Handle missing dependents
df['NumberOfDependents'] = df['NumberOfDependents'].fillna(0)

# Drop helper column
df = df.drop(columns=['age_group'])

print(f"   ✓ Cleaned shape: {df.shape}")
print(f"   ✓ Missing values: {df.isnull().sum().sum()}")


# ═══════════════════════════════════════════════
# STEP 3: FEATURE ENGINEERING
# ═══════════════════════════════════════════════

print("\n✅ STEP 3: FEATURE ENGINEERING...")

# Total delinquency score (WEIGHTED)
df['Total_Delinquency'] = (
    df['NumberOfTime30-59DaysPastDueNotWorse'] * 1 +
    df['NumberOfTime60-89DaysPastDueNotWorse'] * 2 +
    df['NumberOfTimes90DaysLate'] * 3
)

# Income per dependent
df['Income_Per_Dependent'] = df['MonthlyIncome'] / (df['NumberOfDependents'] + 1)

# DTI Ratio
df['DTI_Ratio'] = df['DebtRatio']

# Credit burden
df['Credit_Burden'] = df['NumberOfOpenCreditLinesAndLoans'] / (df['MonthlyIncome'] / 1000 + 1)

print("   ✓ Features engineered:")
print("     - Total_Delinquency (weighted late payments)")
print("     - Income_Per_Dependent")
print("     - DTI_Ratio")
print("     - Credit_Burden")


# ═══════════════════════════════════════════════
# STEP 4: SELECT FEATURES & PREPARE DATA
# ═══════════════════════════════════════════════

print("\n✅ STEP 4: PREPARING DATA...")

# Feature order (MUST match app.py)
feature_names = [
    'Total_Delinquency',
    'RevolvingUtilizationOfUnsecuredLines',
    'NumberOfTime30-59DaysPastDueNotWorse',
    'age',
    'Income_Per_Dependent',
    'MonthlyIncome',
    'DebtRatio',
    'DTI_Ratio',
    'NumberOfOpenCreditLinesAndLoans',
    'Credit_Burden',
    'NumberOfDependents'
]

X = df[feature_names].copy()
y = df['SeriousDlqin2yrs'].copy()

print(f"   ✓ Features selected: {len(feature_names)}")
print(f"   ✓ X shape: {X.shape}")
print(f"   ✓ y shape: {y.shape}")

# Train-Test Split (70-30)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print(f"   ✓ Train: {X_train.shape[0]:,} samples")
print(f"   ✓ Test:  {X_test.shape[0]:,} samples")

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("   ✓ Features scaled")


# ═══════════════════════════════════════════════
# STEP 5: APPLY SMOTE
# ═══════════════════════════════════════════════

print("\n✅ STEP 5: HANDLING CLASS IMBALANCE WITH SMOTE...")

print(f"\n   BEFORE SMOTE:")
print(f"   Good: {(y_train == 0).sum():,} | Bad: {(y_train == 1).sum():,}")
print(f"   Ratio: {(y_train == 0).sum() / (y_train == 1).sum():.1f}:1")

smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)

print(f"\n   AFTER SMOTE:")
print(f"   Good: {(y_train_smote == 0).sum():,} | Bad: {(y_train_smote == 1).sum():,}")
print(f"   Ratio: {(y_train_smote == 0).sum() / (y_train_smote == 1).sum():.1f}:1")


# ═══════════════════════════════════════════════
# STEP 6: TRAIN LOGISTIC REGRESSION
# ═══════════════════════════════════════════════

print("\n✅ STEP 6: TRAINING LOGISTIC REGRESSION...")

model = LogisticRegression(
    max_iter=1000,
    random_state=42,
    class_weight='balanced'
)

model.fit(X_train_smote, y_train_smote)

print("   ✓ Model trained successfully!")


# ═══════════════════════════════════════════════
# STEP 7: EVALUATE MODEL
# ═══════════════════════════════════════════════

print("\n✅ STEP 7: EVALUATING MODEL...")

# Predictions
y_train_pred_proba = model.predict_proba(X_train_scaled)[:, 1]
y_test_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

y_train_pred = model.predict(X_train_scaled)
y_test_pred = model.predict(X_test_scaled)

# AUC-ROC
train_auc = roc_auc_score(y_train, y_train_pred_proba)
test_auc = roc_auc_score(y_test, y_test_pred_proba)

print(f"\n   AUC-ROC Score:")
print(f"   Train: {train_auc:.4f}")
print(f"   Test:  {test_auc:.4f}")

# KS Statistic
def calculate_ks_statistic(y_true, y_pred_proba):
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    ks = max(tpr - fpr)
    return ks

train_ks = calculate_ks_statistic(y_train, y_train_pred_proba)
test_ks = calculate_ks_statistic(y_test, y_test_pred_proba)

print(f"\n   KS Statistic:")
print(f"   Train: {train_ks:.4f}")
print(f"   Test:  {test_ks:.4f}")

# Gini Coefficient
def calculate_gini(y_true, y_pred_proba):
    auc = roc_auc_score(y_true, y_pred_proba)
    gini = 2 * auc - 1
    return gini

test_gini = calculate_gini(y_test, y_test_pred_proba)

print(f"\n   Gini Coefficient: {test_gini:.4f}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_test_pred)
tn, fp, fn, tp = cm.ravel()

print(f"\n   Confusion Matrix:")
print(f"   TN: {tn:,} | FP: {fp:,}")
print(f"   FN: {fn:,} | TP: {tp:,}")


# ═══════════════════════════════════════════════
# STEP 8: SAVE MODEL & SCALER
# ═══════════════════════════════════════════════

print("\n✅ STEP 8: SAVING MODEL & SCALER...")

# Create dictionary with all necessary components
model_data = {
    "model": model,
    "scaler": scaler,
    "feature_names": feature_names,
    "train_means": scaler.mean_,
    "train_stds": scaler.scale_,
    "auc": test_auc,
    "ks": test_ks,
    "gini": test_gini
}

# Save in current directory (notebooks folder)
output_path = 'pd_model_final.pkl'

with open(output_path, 'wb') as f:
    pickle.dump(model_data, f)

print(f"   ✓ Model saved as '{output_path}'")

# Also copy to parent directory for app.py
import os
import shutil

try:
    parent_path = f'../pd_model_final.pkl'
    shutil.copy(output_path, parent_path)
    print(f"   ✓ Also copied to '../pd_model_final.pkl'")
    print(f"   ✓ Full path: {os.path.abspath(parent_path)}")
except Exception as e:
    print(f"   ⚠️ Could not copy to parent directory: {e}")


# ═══════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════

print("\n" + "=" * 70)
print("✅ TRAINING COMPLETE!")
print("=" * 70)

print(f"""
📊 MODEL PERFORMANCE:
   • AUC-ROC:      {test_auc:.4f}
   • KS Statistic: {test_ks:.4f}
   • Gini:         {test_gini:.4f}

📁 FILES SAVED:
   • pd_model_final.pkl (in current directory)
   • ../pd_model_final.pkl (in parent directory)

🎯 FEATURES ({len(feature_names)}):
""")

for i, feat in enumerate(feature_names, 1):
    print(f"   {i:2d}. {feat}")

print(f"""
✨ NEXT STEPS:
   1. Go back to project root: cd ..
   2. Run Streamlit app: streamlit run app.py
   3. Open http://localhost:8501 in browser

🎉 Model ready for deployment!
""")