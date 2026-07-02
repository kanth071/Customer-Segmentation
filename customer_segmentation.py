# ==========================================
# CUSTOMER SEGMENTATION + SEGMENT MODELS
# ==========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import (
    silhouette_score,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    classification_report
)

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

# ==========================================
# 1️⃣ LOAD DATA
# ==========================================

df = pd.read_csv("/content/customer_churn (1).csv")

print("Dataset Shape:", df.shape)
print(df.head())

# ==========================================
# 2️⃣ BASIC PREPROCESSING
# ==========================================

# Drop customer ID if exists
if 'customerID' in df.columns:
    df.drop('customerID', axis=1, inplace=True)

# Encode target
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

# Encode categorical features
for col in df.select_dtypes(include='object').columns:
    df[col] = LabelEncoder().fit_transform(df[col])

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df.drop('Churn', axis=1))

# ==========================================
# 3️⃣ CLUSTERING (2 Algorithms)
# ==========================================

# ---- KMeans ----
kmeans = KMeans(n_clusters=3, random_state=42)
df['KMeans_Segment'] = kmeans.fit_predict(X_scaled)

kmeans_silhouette = silhouette_score(X_scaled, df['KMeans_Segment'])
print("KMeans Silhouette Score:", kmeans_silhouette)

# ---- Hierarchical Clustering ----
agg = AgglomerativeClustering(n_clusters=3)
df['Hierarchical_Segment'] = agg.fit_predict(X_scaled)

agg_silhouette = silhouette_score(X_scaled, df['Hierarchical_Segment'])
print("Hierarchical Silhouette Score:", agg_silhouette)

# We will use KMeans segmentation
df['Segment'] = df['KMeans_Segment']

# ==========================================
# 4️⃣ SEGMENT DISTRIBUTION
# ==========================================

segment_distribution = df['Segment'].value_counts(normalize=True) * 100
print("\nCUSTOMER SEGMENTS:")
for i, perc in segment_distribution.items():
    print(f"Segment {i}: {round(perc,2)}%")

# ==========================================
# 5️⃣ BUILD SEPARATE MODELS FOR EACH SEGMENT
# ==========================================

segment_models = {}
segment_results = {}

for segment in df['Segment'].unique():

    print(f"\n============================")
    print(f"Training Model for Segment {segment}")
    print("============================")

    segment_df = df[df['Segment'] == segment]

    X = segment_df.drop(['Churn', 'Segment', 'KMeans_Segment', 'Hierarchical_Segment'], axis=1)
    y = segment_df['Churn']

    if len(y.unique()) < 2:
        print("Not enough class variation in this segment")
        continue

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # Hyperparameter tuning (RandomForest)
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [5, 10, None],
        'min_samples_split': [2, 5]
    }

    grid = GridSearchCV(
        RandomForestClassifier(random_state=42),
        param_grid,
        cv=3,
        scoring='f1',
        n_jobs=-1
    )

    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_

    y_pred = best_model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    roc = roc_auc_score(y_test, best_model.predict_proba(X_test)[:,1])

    segment_models[segment] = best_model
    segment_results[segment] = {
        "accuracy": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "roc_auc": roc
    }

    print("Best Params:", grid.best_params_)
    print("Accuracy:", acc)
    print("F1 Score:", f1)
    print("ROC AUC:", roc)

# ==========================================
# 6️⃣ FINAL PERFORMANCE SUMMARY
# ==========================================

print("\nMODEL PERFORMANCE SUMMARY:")

for seg, metrics in segment_results.items():
    print(f"\nSegment {seg}:")
    for metric, value in metrics.items():
        print(f"{metric}: {round(value, 3)}")
