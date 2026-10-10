

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GroupShuffleSplit


def locate_dataset_file(data_dir: str) -> str:
    
    candidates = ["ml.csv", "ML.csv"]
    for c in candidates:
        path = os.path.join(data_dir, c)
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"Could not locate ml.csv or ML.csv in {data_dir}")


def inspect_dataset(df: pd.DataFrame) -> dict:
    
    shape = df.shape
    missing_count = int(df.isnull().sum().sum())
    duplicate_full_rows = int(df.duplicated().sum())
    
    # Feature columns (excluding target prognosis and optional medicine)
    excluded = ['prognosis', 'medicine']
    feature_cols = [col for col in df.columns if col not in excluded]
    
    # Constant columns (zero variance)
    constant_columns = [col for col in feature_cols if df[col].nunique() <= 1]
    active_features = [col for col in feature_cols if col not in constant_columns]
    
    # Unique symptom combinations
    duplicate_patterns = int(df.duplicated(subset=active_features).sum())
    unique_patterns_count = len(df.drop_duplicates(subset=active_features))
    
    # Class distribution
    prognosis_counts = df['prognosis'].value_counts().to_dict()
    unique_classes_count = len(prognosis_counts)
    
    # Check consistency: do identical symptom patterns ever map to multiple diseases?
    grouped = df.groupby(active_features)['prognosis'].nunique()
    inconsistent_patterns = int((grouped > 1).sum())

    summary = {
        "total_rows": shape[0],
        "total_columns": shape[1],
        "missing_values": missing_count,
        "duplicate_full_rows": duplicate_full_rows,
        "unique_symptom_patterns": unique_patterns_count,
        "duplicate_symptom_patterns": duplicate_patterns,
        "constant_columns": constant_columns,
        "unique_diseases": unique_classes_count,
        "inconsistent_patterns": inconsistent_patterns,
        "disease_distribution": prognosis_counts,
        "min_samples_per_class": int(df['prognosis'].value_counts().min()),
        "max_samples_per_class": int(df['prognosis'].value_counts().max())
    }
    return summary


def preprocess_data(csv_path: str, test_size: float = 0.2, random_state: int = 42):
    """
    Loads and preprocesses the dataset:
    1. Loads dataset and checks missing values.
    2. Identifies symptom feature columns (excluding 'prognosis' and 'medicine').
    3. Validates strictly binary values (0 and 1).
    4. Applies group-aware train/test splitting based on complete symptom feature vectors
       so that identical symptom combinations are kept strictly together in either train OR test.
    5. Encodes target disease labels with LabelEncoder.
    6. Returns X_train, X_test, y_train, y_test, feature_cols, label_encoder, train_groups, report.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset file not found at: {csv_path}")

    df = pd.read_csv(csv_path)
    report = inspect_dataset(df)

    # 1. Handle missing values if any
    if df.isnull().sum().sum() > 0:
        df = df.dropna()

    # 2. Identify feature columns
    excluded = ['prognosis', 'medicine']
    feature_cols = [c for c in df.columns if c not in excluded]

    # Remove any constant columns if present
    constant_cols = [c for c in feature_cols if df[c].nunique() <= 1]
    feature_cols = [c for c in feature_cols if c not in constant_cols]

    # 3. Validate binary property
    for col in feature_cols:
        unique_vals = set(df[col].dropna().unique())
        if not unique_vals.issubset({0, 1}):
            raise ValueError(f"Feature column '{col}' has non-binary values: {unique_vals}")

    # 4. Group-aware splitting:
    # Assign group ID to each unique symptom combination
    pattern_groups = df.groupby(feature_cols, sort=False).ngroup()

    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(gss.split(df, groups=pattern_groups))

    train_df = df.iloc[train_idx].copy()
    test_df = df.iloc[test_idx].copy()

    X_train = train_df[feature_cols].values
    X_test = test_df[feature_cols].values
    train_groups = pattern_groups.iloc[train_idx].values

    # 5. Encode target labels
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(train_df['prognosis'].values)
    y_test = label_encoder.transform(test_df['prognosis'].values)

    # 6. Audit check: zero pattern overlap
    train_tuples = set(map(tuple, X_train))
    test_tuples = set(map(tuple, X_test))
    overlap = len(train_tuples.intersection(test_tuples))
    assert overlap == 0, f"Critical: Data leakage detected! {overlap} overlapping patterns between train and test."

    report["final_feature_count"] = len(feature_cols)
    report["final_feature_columns"] = feature_cols
    report["train_samples"] = len(X_train)
    report["test_samples"] = len(X_test)
    report["pattern_overlap"] = overlap
    report["train_classes_count"] = len(np.unique(y_train))
    report["test_classes_count"] = len(np.unique(y_test))

    return X_train, X_test, y_train, y_test, feature_cols, label_encoder, train_groups, report


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, "data")
    data_path = locate_dataset_file(data_dir)
    
    print("=" * 70)
    print(" NEW DATASET INSPECTION & GROUP-AWARE SPLIT AUDIT")
    print("=" * 70)
    print(f"Reading dataset: {data_path}")
    
    X_tr, X_te, y_tr, y_te, feats, le, grps, rep = preprocess_data(data_path)
    
    print(f"\n[1] Basic Statistics:")
    print(f"    - Total records (rows): {rep['total_rows']}")
    print(f"    - Total columns: {rep['total_columns']}")
    print(f"    - Missing values: {rep['missing_values']}")
    print(f"    - Exact duplicate full rows: {rep['duplicate_full_rows']}")
    print(f"    - Unique symptom pattern combinations: {rep['unique_symptom_patterns']}")
    print(f"    - Symptom pattern duplicates: {rep['duplicate_symptom_patterns']}")
    print(f"    - Active binary symptom features: {rep['final_feature_count']}")
    
    print(f"\n[2] Target Disease Distribution:")
    print(f"    - Total disease classes: {rep['unique_diseases']}")
    print(f"    - Min samples per class: {rep['min_samples_per_class']}")
    print(f"    - Max samples per class: {rep['max_samples_per_class']}")
    print(f"    - Symptom patterns with multiple/conflicting prognoses: {rep['inconsistent_patterns']}")
    
    print(f"\n[3] Leakage-Free Train/Test Split Audit:")
    print(f"    - Training samples: {rep['train_samples']}")
    print(f"    - Testing samples: {rep['test_samples']}")
    print(f"    - Train class coverage: {rep['train_classes_count']} / {rep['unique_diseases']}")
    print(f"    - Test class coverage: {rep['test_classes_count']} / {rep['unique_diseases']}")
    print(f"    - Pattern overlap between Train and Test: {rep['pattern_overlap']} (Zero Leakage)")
    print("=" * 70)
