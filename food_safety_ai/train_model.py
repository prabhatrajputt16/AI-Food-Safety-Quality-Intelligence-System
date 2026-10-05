"""
AI Food Safety & Quality Intelligence System
XGBoost Model Trainer
Trains TWO models:
  1. quality_grade   (A/B/C/D)   — multi-class
  2. safety_status   (Safe/Monitor/Unsafe) — multi-class
"""

import os, warnings, joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing   import LabelEncoder, StandardScaler
from sklearn.metrics         import (
    accuracy_score, f1_score, recall_score, precision_score,
    classification_report, confusion_matrix, roc_auc_score
)
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
BASE = os.path.dirname(os.path.abspath(__file__))
os.makedirs(os.path.join(BASE, "models"), exist_ok=True)
os.makedirs(os.path.join(BASE, "plots"),  exist_ok=True)

# ── 1. Load data ──────────────────────────────────────────────────────────────
df = pd.read_csv(os.path.join(BASE, "food_safety_dataset.csv"))
print(f"Loaded {len(df)} rows")

CAT_COLS = ["food_category", "storage_type", "packaging_type", "season", "supplier",
            "prediction_reason"]

# Encode categoricals
label_encoders = {}
for col in CAT_COLS:
    le = LabelEncoder()
    df[col + "_enc"] = le.fit_transform(df[col])
    label_encoders[col] = le

# Feature columns (numeric + encoded)
NUM_COLS = [
    "temperature", "humidity", "storage_days", "pH", "moisture_percent",
    "defect_count", "packaging_damage", "odor_score", "color_uniformity",
    "texture_score", "microbial_load", "water_activity", "fat_content",
    "protein_content", "sugar_content", "salt_content", "atp_reading",
    "ethylene_ppm", "co2_percent", "batch_size", "transport_hours",
    "cold_chain_break", "freshness_index", "risk_score", "quality_score",
    "shelf_life_days",
]
ENC_COLS = [c + "_enc" for c in CAT_COLS]
FEATURE_COLS = NUM_COLS + ENC_COLS

X = df[FEATURE_COLS].copy()

# Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ── Target encoders ───────────────────────────────────────────────────────────
le_quality = LabelEncoder()
le_safety  = LabelEncoder()
y_quality  = le_quality.fit_transform(df["quality_grade"])   # A=0 B=1 C=2 D=3
y_safety   = le_safety.fit_transform(df["safety_status"])    # Monitor=0 Safe=1 Unsafe=2

# ── 2. Train/test split ───────────────────────────────────────────────────────
X_tr, X_te, yq_tr, yq_te = train_test_split(X_scaled, y_quality, test_size=0.2, random_state=42, stratify=y_quality)
_,    _,    ys_tr, ys_te = train_test_split(X_scaled, y_safety,  test_size=0.2, random_state=42, stratify=y_safety)

# ── 3. XGBoost hyperparams ────────────────────────────────────────────────────
COMMON_PARAMS = dict(
    n_estimators      = 600,
    max_depth         = 7,
    learning_rate     = 0.05,
    subsample         = 0.85,
    colsample_bytree  = 0.85,
    reg_alpha         = 0.1,
    reg_lambda        = 1.5,
    min_child_weight  = 3,
    gamma             = 0.1,
    use_label_encoder = False,
    eval_metric       = "mlogloss",
    random_state      = 42,
    n_jobs            = -1,
)

# ── 4. Quality Grade model ────────────────────────────────────────────────────
print("\n-- Training Quality Grade model --")
model_q = XGBClassifier(num_class=4, objective="multi:softmax", **COMMON_PARAMS)
model_q.fit(X_tr, yq_tr,
            eval_set=[(X_te, yq_te)],
            verbose=False)

yq_pred = model_q.predict(X_te)
yq_prob = model_q.predict_proba(X_te)

acc_q   = accuracy_score(yq_te, yq_pred)
f1_q    = f1_score(yq_te, yq_pred, average="weighted")
rec_q   = recall_score(yq_te, yq_pred, average="weighted")
pre_q   = precision_score(yq_te, yq_pred, average="weighted")
try:
    auc_q = roc_auc_score(yq_te, yq_prob, multi_class="ovr", average="weighted")
except Exception:
    auc_q = 0

print(f"  Accuracy  : {acc_q:.4f}")
print(f"  F1 (wt)   : {f1_q:.4f}")
print(f"  Recall    : {rec_q:.4f}")
print(f"  Precision : {pre_q:.4f}")
print(f"  ROC-AUC   : {auc_q:.4f}")
print(classification_report(yq_te, yq_pred, target_names=le_quality.classes_))

# CV
cv_q = cross_val_score(
    XGBClassifier(num_class=4, objective="multi:softmax", **COMMON_PARAMS),
    X_scaled, y_quality, cv=StratifiedKFold(5), scoring="f1_weighted", n_jobs=-1
)
print(f"  CV F1 (5-fold): {cv_q.mean():.4f} +/- {cv_q.std():.4f}")

# ── 5. Safety Status model ────────────────────────────────────────────────────
print("\n-- Training Safety Status model --")
model_s = XGBClassifier(num_class=3, objective="multi:softmax", **COMMON_PARAMS)
model_s.fit(X_tr, ys_tr,
            eval_set=[(X_te, ys_te)],
            verbose=False)

ys_pred = model_s.predict(X_te)
ys_prob = model_s.predict_proba(X_te)

acc_s   = accuracy_score(ys_te, ys_pred)
f1_s    = f1_score(ys_te, ys_pred, average="weighted")
rec_s   = recall_score(ys_te, ys_pred, average="weighted")
pre_s   = precision_score(ys_te, ys_pred, average="weighted")
try:
    auc_s = roc_auc_score(ys_te, ys_prob, multi_class="ovr", average="weighted")
except Exception:
    auc_s = 0

print(f"  Accuracy  : {acc_s:.4f}")
print(f"  F1 (wt)   : {f1_s:.4f}")
print(f"  Recall    : {rec_s:.4f}")
print(f"  Precision : {pre_s:.4f}")
print(f"  ROC-AUC   : {auc_s:.4f}")
print(classification_report(ys_te, ys_pred, target_names=le_safety.classes_))

cv_s = cross_val_score(
    XGBClassifier(num_class=3, objective="multi:softmax", **COMMON_PARAMS),
    X_scaled, y_safety, cv=StratifiedKFold(5), scoring="f1_weighted", n_jobs=-1
)
print(f"  CV F1 (5-fold): {cv_s.mean():.4f} +/- {cv_s.std():.4f}")

# ── 6. Confusion matrices ─────────────────────────────────────────────────────
def save_cm(y_true, y_pred, labels, title, fname):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels, ax=ax,
                linewidths=0.5, linecolor="white")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("Actual",    fontsize=11)
    plt.tight_layout()
    plt.savefig(fname, dpi=150)
    plt.close()

save_cm(yq_te, yq_pred, le_quality.classes_,
        "Confusion Matrix — Quality Grade",
        os.path.join(BASE, "plots", "cm_quality.png"))

save_cm(ys_te, ys_pred, le_safety.classes_,
        "Confusion Matrix — Safety Status",
        os.path.join(BASE, "plots", "cm_safety.png"))

# ── 7. Feature importance ─────────────────────────────────────────────────────
def save_fi(model, title, fname):
    fi = pd.Series(model.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False).head(20)
    fig, ax = plt.subplots(figsize=(9, 6))
    fi.sort_values().plot(kind="barh", ax=ax, color="#2563EB")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Importance")
    plt.tight_layout()
    plt.savefig(fname, dpi=150)
    plt.close()

save_fi(model_q, "Top-20 Feature Importance — Quality Grade",
        os.path.join(BASE, "plots", "fi_quality.png"))
save_fi(model_s, "Top-20 Feature Importance — Safety Status",
        os.path.join(BASE, "plots", "fi_safety.png"))

# ── 8. Save artefacts ─────────────────────────────────────────────────────────
joblib.dump(model_q,        os.path.join(BASE, "models", "model_quality.pkl"))
joblib.dump(model_s,        os.path.join(BASE, "models", "model_safety.pkl"))
joblib.dump(scaler,         os.path.join(BASE, "models", "scaler.pkl"))
joblib.dump(label_encoders, os.path.join(BASE, "models", "label_encoders.pkl"))
joblib.dump(le_quality,     os.path.join(BASE, "models", "le_quality.pkl"))
joblib.dump(le_safety,      os.path.join(BASE, "models", "le_safety.pkl"))
joblib.dump(FEATURE_COLS,   os.path.join(BASE, "models", "feature_cols.pkl"))

# Save metrics for display in UI
metrics = {
    "quality": {
        "accuracy":  round(acc_q,  4),
        "f1":        round(f1_q,   4),
        "recall":    round(rec_q,  4),
        "precision": round(pre_q,  4),
        "auc":       round(auc_q,  4),
        "cv_f1_mean": round(cv_q.mean(), 4),
        "cv_f1_std":  round(cv_q.std(),  4),
    },
    "safety": {
        "accuracy":  round(acc_s,  4),
        "f1":        round(f1_s,   4),
        "recall":    round(rec_s,  4),
        "precision": round(pre_s,  4),
        "auc":       round(auc_s,  4),
        "cv_f1_mean": round(cv_s.mean(), 4),
        "cv_f1_std":  round(cv_s.std(),  4),
    },
}
import json
with open(os.path.join(BASE, "models", "metrics.json"), "w") as f:
    json.dump(metrics, f, indent=2)

print("\nAll artefacts saved to food_safety_ai/models/")
