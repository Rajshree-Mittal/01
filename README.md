# Balancing_IOT

Balancing_IOT is a professional IoT threat classification project that tackles extreme class imbalance using a structured preprocessing and ensemble modeling pipeline.

## Project Overview

This repository demonstrates a robust workflow for preparing IoT datasets, generating balanced training data, and training specialized classifiers to improve detection of rare attack types.

Key concepts used:

- Data preprocessing and feature engineering for network traffic datasets
- Class imbalance handling with augmentation and sample weighting
- Multi-stage classification using a main model plus specialized confusion-cluster models
- Model evaluation with accuracy, recall, confusion matrices, and classification reports

## Architecture

1. Data ingestion
   - Read and merge source IoT datasets from CSV files
   - Detect label columns automatically for binary and multiclass classification

2. Feature preprocessing
   - Remove identifiers and irrelevant metadata
   - Keep only numeric features
   - Scale features with standardization

3. Balancing and augmentation
   - Apply synthetic sampling for binary label balancing
   - Use PIDA-style augmentation to enrich minority attack classes

4. Model training and evaluation
   - Train a baseline XGBoost multiclass model
   - Compute confusion clusters from the baseline predictions
   - Apply dynamic class weights based on class recall
   - Train a weighted main model and specialized models for confusing class groups
   - Combine predictions with a confidence-based multi-stage decision process

5. Output and persistence
   - Print final accuracy, classification report, and confusion matrix
   - Save trained models to disk for later use

## Nomenclature

- `PIDA` — the augmentation approach used to generate synthetic minority-class samples and improve representation of rare attack types.
- `binary_label` — a binary label used to distinguish normal traffic from malicious traffic.
- `multi_label` — a multiclass label used to distinguish between specific attack categories.
- `main model` — the weighted XGBoost model trained on the full multiclass dataset.
- `confusion clusters` — groups of classes with high mutual misclassification rates, each handled by a specialized secondary model.
- `dynamic weighting` — the sample weight strategy calculated from per-class recall to emphasize underperforming classes.

## Primary files

- `processing.py` — dataset loading, preprocessing, label creation, balancing, and augmentation logic
- `final_work.py` — XGBoost training pipeline, confusion cluster detection, weighted modeling, and multi-stage prediction

## Notes

This project is designed for IoT intrusion detection research and is structured to support experiments with different datasets and imbalance mitigation techniques.
