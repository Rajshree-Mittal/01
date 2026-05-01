# =========================================================
# 0  IMPORTS
# =========================================================

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    confusion_matrix,
    recall_score,
)

from xgboost import XGBClassifier
from collections import Counter
import joblib


# =========================================================
# 1  LOAD DATA
#    Read the pre-processed ToN-IoT CSV and separate the
#    features from both label columns.
# =========================================================

DATA_PATH = 'processed_datasets/ton-iot_final.csv'

df = pd.read_csv(DATA_PATH)

print("Dataset shape:", df.shape)

X = df.drop(columns=["multi_label", "binary_label"])
y = df["multi_label"]

print("\nMulticlass distribution:", Counter(y))


# =========================================================
# 2  TRAIN / TEST SPLIT
#    Stratified split so every class is represented
#    proportionally in both halves.
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# =========================================================
# 3  BINARY CLASSIFICATION
#    A quick normal-vs-attack sanity check before we dive
#    into the fine-grained multiclass problem.
# =========================================================

print("\n==============================")
print("BINARY CLASSIFICATION")
print("==============================")

# Prepare binary labels
yb = df["binary_label"]

# Split for binary classification
Xb_train, Xb_test, yb_train, yb_test = train_test_split(
    X, yb,
    test_size=0.2,
    random_state=42,
    stratify=yb
)

binary_model = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    use_label_encoder=False,
    eval_metric="logloss"
)

binary_model.fit(Xb_train, yb_train)

yb_pred = binary_model.predict(Xb_test)

print("\nBinary Accuracy:", accuracy_score(yb_test, yb_pred))
print("\nBinary Classification Report:\n")
print(classification_report(yb_test, yb_pred))


# =========================================================
# 4  BASE MODEL
#    A moderately deep XGBoost trained on all classes with
#    no class weighting — this is our starting benchmark.
# =========================================================

print("\n==============================")
print("BASE MODEL")
print("==============================")

base_model = XGBClassifier(
    n_estimators=400,
    max_depth=10,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="multi:softprob",
    num_class=len(np.unique(y)),
    n_jobs=-1
)

base_model.fit(X_train, y_train)

base_pred = base_model.predict(X_test)

print("\nBase Accuracy:", accuracy_score(y_test, base_pred))

cm = confusion_matrix(y_test, base_pred)
print("\nConfusion Matrix:\n", cm)


# =========================================================
# 5  DETECT CONFUSION CLUSTERS
#    Scan the confusion matrix for classes that bleed into
#    each other above a 5 % error threshold.  These become
#    the targets for our specialised sub-models later.
# =========================================================

print("\n==============================")
print("DETECTING CONFUSION CLUSTERS")
print("==============================")

threshold = 0.05
clusters = []
n_classes = cm.shape[0]

for i in range(n_classes):
    confused = []
    total = np.sum(cm[i])

    for j in range(n_classes):
        if i != j:
            ratio = cm[i][j] / total
            if ratio > threshold:
                confused.append(j)

    if len(confused) > 0:
        cluster = list(set([i] + confused))
        clusters.append(cluster)

# remove duplicates
unique_clusters = []
for c in clusters:
    if c not in unique_clusters:
        unique_clusters.append(c)

print("\nDetected Clusters:")
for c in unique_clusters:
    print(c)


# =========================================================
# 6  DYNAMIC CLASS WEIGHTING
#    Classes the base model struggles with (low recall) get
#    higher sample weights so the main model pays them more
#    attention during training.
# =========================================================

recalls = recall_score(y_test, base_pred, average=None)

epsilon = 1e-6
classes = np.unique(y_train)

class_weights = {
    cls: 1 / (recalls[i] + epsilon)
    for i, cls in enumerate(classes)
}

# normalize + clip
total = sum(class_weights.values())
class_weights = {k: min(v / total, 5.0) for k, v in class_weights.items()}

sample_weights = y_train.map(class_weights).values

print("\nClass Weights:", class_weights)


# =========================================================
# 7  MAIN MODEL  (WEIGHTED)
#    A deeper, longer-trained model using the per-sample
#    weights derived above.  This is our primary predictor.
# =========================================================

print("\n==============================")
print("MAIN MODEL (WEIGHTED)")
print("==============================")

main_model = XGBClassifier(
    n_estimators=500,
    max_depth=12,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="multi:softprob",
    num_class=len(classes),
    n_jobs=-1
)

main_model.fit(X_train, y_train, sample_weight=sample_weights)

main_pred = main_model.predict(X_test)

print("\nWeighted Accuracy:", accuracy_score(y_test, main_pred))


# =========================================================
# 8  TRAIN CONFUSION MODELS
#    For each confusion cluster, train a focused XGBoost
#    that only sees samples from those classes.  Labels are
#    remapped to 0…N so XGBoost stays happy, and we keep
#    the reverse mapping to restore original class IDs.
# =========================================================

print("\n==============================")
print("TRAINING CONFUSION MODELS")
print("==============================")

conf_models = {}

for cluster in unique_clusters:

    if len(cluster) < 2:
        continue

    print(f"\nCluster: {cluster}")

    mask = y_train.isin(cluster)

    X_conf = X_train[mask]
    y_conf = y_train[mask]

    # 🔥 FIX: remap labels
    unique_classes = sorted(cluster)
    class_to_idx = {c: i for i, c in enumerate(unique_classes)}
    idx_to_class = {i: c for c, i in class_to_idx.items()}

    y_conf_mapped = y_conf.map(class_to_idx)

    model = XGBClassifier(
        n_estimators=300,
        max_depth=8,
        learning_rate=0.05,
        objective="multi:softprob",
        num_class=len(unique_classes),
        n_jobs=-1
    )

    model.fit(X_conf, y_conf_mapped)

    # store both model and mapping
    conf_models[tuple(cluster)] = (model, idx_to_class)


# =========================================================
# 9  FINAL PREDICTION  (MULTI-STAGE)
#    Start with the main model's predictions.  Wherever it
#    is uncertain (confidence < 80 %) AND the predicted
#    class belongs to a confusion cluster, hand that sample
#    off to the corresponding specialised model instead.
# =========================================================

print("\n==============================")
print("FINAL PREDICTION")
print("==============================")

final_pred = main_model.predict(X_test)
probs_all = main_model.predict_proba(X_test)

CONF_THRESHOLD = 0.8   

for i in range(len(final_pred)):

    confidence = np.max(probs_all[i])

    for cluster, (model, idx_to_class) in conf_models.items():

        # apply confusion model ONLY if:
        # 1. prediction is in cluster
        # 2. model is not confident
        if final_pred[i] in cluster and confidence < CONF_THRESHOLD:

            pred = model.predict(X_test.iloc[i:i+1])
            pred = int(pred[0])

            final_pred[i] = idx_to_class[pred]
            break


# =========================================================
# 10  FINAL RESULTS
#     Print the end-to-end evaluation of the full
#     multi-stage pipeline.
# =========================================================

print("\n==============================")
print("FINAL RESULTS")
print("==============================")

print("\nFinal Accuracy:", accuracy_score(y_test, final_pred))

print("\nClassification Report:\n")
print(classification_report(y_test, final_pred))


print("\nConfusion Matrix:\n")
print(confusion_matrix(y_test, final_pred))


# =========================================================
# 11  SAVE MODELS
#     Persist the main model and all confusion sub-models
#     so they can be loaded for inference without retraining.
# =========================================================

joblib.dump(main_model, "main_xgb_model.pkl")
joblib.dump(conf_models, "confusion_models.pkl")

print("\nModels saved successfully!")