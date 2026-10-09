"""
AI-Based Disease Prediction System - Model Training & Evaluation Pipeline

Trains, rigorously evaluates, and benchmarks six classification algorithms:
1. Decision Tree
2. Random Forest
3. K-Nearest Neighbors (KNN)
4. Support Vector Machine (SVM)
5. Logistic Regression
6. Naive Bayes

Trained on the updated preprocessed dataset (38,434 records, 165 symptoms, 42 diseases)
with group-aware train/test splitting and group-aware cross-validation.
"""

import os
import json
import warnings
import joblib
import numpy as np
from sklearn.model_selection import GroupKFold, cross_validate
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from preprocessing import preprocess_data, locate_dataset_file

# Suppress minor convergence warnings
warnings.filterwarnings('ignore')


def train_and_compare_models(data_path: str, output_dir: str):
    """
    Executes the training, evaluation, and artifact export pipeline.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Preprocess data with group-aware zero-leakage pattern splitting
    X_train, X_test, y_train, y_test, feature_cols, label_encoder, train_groups, dataset_report = preprocess_data(
        data_path, test_size=0.2, random_state=42
    )

    # Verify zero overlap between train and test patterns
    train_patterns = set(map(tuple, X_train))
    test_patterns = set(map(tuple, X_test))
    overlap_count = len(train_patterns.intersection(test_patterns))
    assert overlap_count == 0, f"Critical: Data leakage detected! {overlap_count} overlapping patterns."

    # 2. Define the 6 models with fixed random_state for strict reproducibility
    models = {
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        "KNN": KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
        "SVM": CalibratedClassifierCV(LinearSVC(random_state=42, max_iter=2000), cv=3),
        "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
        "Naive Bayes": MultinomialNB()
    }

    # Group-aware cross-validation on training partition only (3 folds)
    gkf = GroupKFold(n_splits=3)

    results = []
    trained_models = {}

    print("\n" + "=" * 90)
    print(" RIGOROUS MODEL TRAINING & BENCHMARKING (NEW DATASET: 38,434 RECORDS, 42 DISEASES)")
    print("=" * 90)

    for name, model in models.items():
        print(f"--> Training and cross-validating: {name}...")
        # Group-aware cross-validation on training data only
        cv_scores = cross_validate(
            model,
            X_train,
            y_train,
            groups=train_groups,
            cv=gkf,
            scoring=['accuracy', 'f1_weighted'],
            return_train_score=False
        )
        cv_acc_mean = float(np.mean(cv_scores['test_accuracy']))
        cv_f1_mean = float(np.mean(cv_scores['test_f1_weighted']))

        # Fit model on training set
        model.fit(X_train, y_train)
        trained_models[name] = model

        # Evaluate on Training set
        y_train_pred = model.predict(X_train)
        train_acc = float(accuracy_score(y_train, y_train_pred))
        train_f1 = float(f1_score(y_train, y_train_pred, average='weighted', zero_division=0))

        # Evaluate on Untouched Held-out Test set
        y_test_pred = model.predict(X_test)
        test_acc = float(accuracy_score(y_test, y_test_pred))
        test_prec = float(precision_score(y_test, y_test_pred, average='weighted', zero_division=0))
        test_rec = float(recall_score(y_test, y_test_pred, average='weighted', zero_division=0))
        test_f1 = float(f1_score(y_test, y_test_pred, average='weighted', zero_division=0))
        cm = confusion_matrix(y_test, y_test_pred).tolist()

        model_stat = {
            "model_name": name,
            "train_accuracy": round(train_acc, 4),
            "train_f1_score": round(train_f1, 4),
            "cv_accuracy": round(cv_acc_mean, 4),
            "cv_f1_score": round(cv_f1_mean, 4),
            "test_accuracy": round(test_acc, 4),
            "test_precision": round(test_prec, 4),
            "test_recall": round(test_rec, 4),
            "test_f1_score": round(test_f1, 4),
            "confusion_matrix": cm,
            "supports_probability": hasattr(model, "predict_proba")
        }
        results.append(model_stat)

    # 3. Model Selection: STRICTLY based on Cross-Validation on Training data
    # (Never using the held-out test set for selection)
    results_sorted = sorted(
        results,
        key=lambda x: (x["cv_accuracy"], x["cv_f1_score"]),
        reverse=True
    )

    best_stat = results_sorted[0]
    best_name = best_stat["model_name"]
    best_model = trained_models[best_name]

    selection_reason = (
        f"Selected '{best_name}' based strictly on training-set group-aware cross-validation: "
        f"Highest mean CV accuracy ({best_stat['cv_accuracy'] * 100:.2f}%) and weighted CV F1-score "
        f"({best_stat['cv_f1_score'] * 100:.2f}%). Held-out test accuracy is {best_stat['test_accuracy'] * 100:.2f}%."
    )

    # 4. Save artifacts
    best_model_path = os.path.join(output_dir, "best_model.pkl")
    label_encoder_path = os.path.join(output_dir, "label_encoder.pkl")
    feature_columns_path = os.path.join(output_dir, "feature_columns.pkl")
    results_json_path = os.path.join(output_dir, "model_results.json")

    joblib.dump(best_model, best_model_path)
    joblib.dump(label_encoder, label_encoder_path)
    joblib.dump(feature_cols, feature_columns_path)

    export_payload = {
        "dataset_version": "v2.0 (preprocessed ml.csv)",
        "best_model_name": best_name,
        "selection_reason": selection_reason,
        "best_model_test_metrics": {
            "train_accuracy": best_stat["train_accuracy"],
            "cv_accuracy": best_stat["cv_accuracy"],
            "test_accuracy": best_stat["test_accuracy"],
            "test_precision": best_stat["test_precision"],
            "test_recall": best_stat["test_recall"],
            "test_f1_score": best_stat["test_f1_score"]
        },
        "models": results,
        "classes": list(label_encoder.classes_),
        "dataset_summary": {
            "total_rows": dataset_report["total_rows"],
            "duplicate_rows": dataset_report["duplicate_full_rows"],
            "unique_patterns": dataset_report["unique_symptom_patterns"],
            "features_count": dataset_report["final_feature_count"],
            "diseases_count": dataset_report["unique_diseases"],
            "train_samples": dataset_report["train_samples"],
            "test_samples": dataset_report["test_samples"],
            "pattern_overlap": overlap_count,
            "min_samples_per_class": dataset_report["min_samples_per_class"],
            "max_samples_per_class": dataset_report["max_samples_per_class"],
            "inconsistent_patterns": dataset_report["inconsistent_patterns"]
        }
    }

    with open(results_json_path, "w", encoding="utf-8") as f:
        json.dump(export_payload, f, indent=2)

    # 5. Print Comparison Table
    print("\n" + "=" * 95)
    print(" MODEL PERFORMANCE COMPARISON TABLE (NEW DATASET)")
    print("=" * 95)
    print(f"| {'Model':<20} | {'Train Acc':<9} | {'CV Acc':<8} | {'Test Acc':<8} | {'Precision':<9} | {'Recall':<8} | {'F1-score':<8} |")
    print("|" + "-"*22 + "|" + "-"*11 + "|" + "-"*10 + "|" + "-"*10 + "|" + "-"*11 + "|" + "-"*10 + "|" + "-"*10 + "|")
    for r in results:
        print(f"| {r['model_name']:<20} | {r['train_accuracy']*100:<8.2f}% | {r['cv_accuracy']*100:<7.2f}% | {r['test_accuracy']*100:<7.2f}% | {r['test_precision']*100:<8.2f}% | {r['test_recall']*100:<7.2f}% | {r['test_f1_score']*100:<7.2f}% |")
    print("=" * 95)
    print(f"\n>> BEST MODEL SELECTED (Via CV): {best_name}")
    print(f">> SELECTION RATIONALE: {selection_reason}")
    print(f">> PATTERN OVERLAP (TRAIN vs TEST): {overlap_count} (Confirmed Zero Leakage)")
    print(f">> SAVED ARTIFACTS TO: {output_dir}")
    print("=" * 95)

    return export_payload


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data")
    data_file = locate_dataset_file(data_dir)
    models_dir = os.path.join(base_dir, "models")
    train_and_compare_models(data_file, models_dir)
