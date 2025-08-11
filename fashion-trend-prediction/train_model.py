import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

try:
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier
    from imblearn.over_sampling import SMOTE
except ImportError:
    print('Please install xgboost, lightgbm, and imbalanced-learn.')
    exit(1)

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# Load the data
print("Loading fashion trends data...")
data_path = os.path.join('data', 'processed', 'fashion_trends_clean.csv')
df = pd.read_csv(data_path)

print(f"Data loaded: {len(df)} records")

# Data preprocessing (matching your notebook)
better_df = df.copy()

# Impute missing values for important features
for col in ['popularity_change', 'trend_momentum']:
    if col in better_df.columns:
        median_val = better_df[col].median()
        better_df[col] = better_df[col].fillna(median_val)
        print(f"Imputed missing values in {col} with median: {median_val}")

# Cap outliers
from scipy.stats import zscore
num_cols = better_df.select_dtypes(include=np.number).columns
for col in num_cols:
    if col in better_df.columns:
        z = np.abs(zscore(better_df[col].fillna(0)))
        cap_val = better_df[col].mean() + 3 * better_df[col].std()
        floor_val = better_df[col].mean() - 3 * better_df[col].std()
        outliers = (z > 3)
        n_outliers = outliers.sum()
        if n_outliers > 0:
            better_df.loc[better_df[col] > cap_val, col] = cap_val
            better_df.loc[better_df[col] < floor_val, col] = floor_val

# Add interaction feature
if 'popularity_7day_avg' in better_df.columns and 'trend_momentum' in better_df.columns:
    better_df['pop7_trendmom'] = better_df['popularity_7day_avg'] * better_df['trend_momentum']
    print("Added interaction feature: pop7_trendmom")

df = better_df.copy()

# Identify columns
date_col = [col for col in df.columns if 'date' in col.lower()][0]
cat_col = [col for col in df.columns if 'category' in col.lower()][0]
score_col = [col for col in df.columns if 'score' in col.lower() or 'trend' in col.lower()][0]

print(f"Using columns: date={date_col}, category={cat_col}, score={score_col}")

# Feature engineering
df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
df['year'] = df[date_col].dt.year
df['month'] = df[date_col].dt.month
df['day'] = df[date_col].dt.day

# Select features
def get_features(df):
    features = ['year', 'month', 'day']
    for col in ['region', 'location', 'country', 'item', 'popularity_7day_avg', 'popularity_30day_avg', 'popularity_change', 'trend_momentum', 'search_volume', 'pop7_trendmom']:
        if col in df.columns:
            features.append(col)
    return features

features = get_features(df)
print(f"Selected features: {features}")

# One-hot encode category
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
cat_encoded = encoder.fit_transform(df[[cat_col]])
cat_encoded_df = pd.DataFrame(cat_encoded, columns=encoder.get_feature_names_out([cat_col]), index=df.index)

X = pd.concat([df[features], cat_encoded_df], axis=1).fillna(0)
median_score = df[score_col].median()
y = (df[score_col] > median_score).astype(int)

print(f"Features shape: {X.shape}, Target shape: {y.shape}")
print(f"Class distribution: {y.value_counts().to_dict()}")

# Balance classes if needed
if abs(y.mean() - 0.5) > 0.1 and X.shape[0] > 100:
    smote = SMOTE(random_state=42)
    X, y = smote.fit_resample(X, y)
    print(f"After SMOTE: {X.shape}, {y.shape}")

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Ensure only numeric columns are passed to the scaler
X_train = pd.DataFrame(X_train).select_dtypes(include=[np.number])
X_test = pd.DataFrame(X_test).select_dtypes(include=[np.number])

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Training models...")

# Define models
models = {
    'XGBoost': XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'),
    'LightGBM': LGBMClassifier(random_state=42),
    'RandomForest': RandomForestClassifier(random_state=42),
    'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42)
}

results = []
trained_models = {}

for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred) * 100
    prec = precision_score(y_test, y_pred) * 100
    rec = recall_score(y_test, y_pred) * 100
    f1 = f1_score(y_test, y_pred) * 100
    
    results.append({'Model': name, 'Accuracy (%)': acc, 'Precision (%)': prec, 'Recall (%)': rec, 'F1 (%)': f1})
    trained_models[name] = model
    
    print(f"{name} - Accuracy: {acc:.2f}%, Precision: {prec:.2f}%, Recall: {rec:.2f}%, F1: {f1:.2f}%")

results_df = pd.DataFrame(results)
results_df = results_df.sort_values('Accuracy (%)', ascending=False)
print("\nModel Accuracy Table:")
print(results_df.round(2))

# Get the best model
best_model_name = results_df.iloc[0]['Model']
best_model = trained_models[best_model_name]
best_accuracy = results_df.iloc[0]['Accuracy (%)']

print(f"\nBest model: {best_model_name} with {best_accuracy:.2f}% accuracy")

# Save the best model and preprocessing objects
model_data = {
    'model': best_model,
    'scaler': scaler,
    'encoder': encoder,
    'feature_names': list(X_train.columns),
    'categorical_features': [cat_col],
    'date_column': date_col,
    'score_column': score_col,
    'median_score': median_score,
    'model_performance': {
        'accuracy': results_df.iloc[0]['Accuracy (%)'],
        'precision': results_df.iloc[0]['Precision (%)'],
        'recall': results_df.iloc[0]['Recall (%)'],
        'f1_score': results_df.iloc[0]['F1 (%)']
    },
    'model_name': best_model_name
}

# Create models directory if it doesn't exist
os.makedirs('models', exist_ok=True)

# Save the model
model_path = os.path.join('models', 'best_fashion_trend_model.pkl')
with open(model_path, 'wb') as f:
    pickle.dump(model_data, f)

print(f"Model saved to {model_path}")
print("Model training complete!")
