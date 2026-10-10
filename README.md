# Smart Disease Prediction System Using Ensembling

A complete, beginner-friendly college project that predicts possible diseases based on user-selected symptoms using supervised multiclass machine learning with ensemble techniques. The system features a **React (Vite + Tailwind CSS)** frontend, a **FastAPI (Python)** REST backend, a locally trained **Scikit-learn** classification engine (comparing 8 distinct algorithms across Bagging, Voting, and Stacking ensembles), and an **SQLite** database for storing prediction history.

> **Medical Disclaimer:** This system is developed strictly for educational and academic demonstration purposes. It does not provide clinical medical diagnoses, prescription recommendations, or treatment plans. Always consult a licensed healthcare professional for medical concerns.

---

## 1. System Architecture & Features

- **Dynamic Symptom Selection:** 165 standardized symptoms loaded dynamically from the dataset (`backend/data/ml.csv`) with real-time search filtering, category-free chip toggles, and interactive tags.
- **Zero-Leakage Group-Aware Splitting:** Splitting performed across unique symptom combinations (`GroupShuffleSplit` and `GroupKFold`) ensuring **0% pattern overlap** between training and testing sets.
- **Comprehensive 8-Model Benchmark Suite:**
  - **Individual Base Classifiers:** Decision Tree, K-Nearest Neighbors (KNN), Linear Support Vector Machine (with Platt probability calibration), Logistic Regression, Multinomial Naive Bayes.
  - **Bagging Ensemble:** Random Forest (100 parallel decision trees with bootstrap aggregation).
  - **Voting Ensemble:** Soft Voting Classifier combining Random Forest, Calibrated Linear SVM, and Logistic Regression with weighted average probabilities.
  - **Stacking Ensemble:** Stacking Classifier using Random Forest, Calibrated Linear SVM, and Logistic Regression as base estimators with Logistic Regression meta-learner and leakage-safe inner `GroupKFold` cross-validation.
- **Differential Disease Ranking:** Outputs the primary predicted disease along with top 3 differential candidates and relative statistical probability scores.
- **Clinical Safety Rules & Uncertain Prediction Handling:**
  - Checks for urgent red-flag symptoms (e.g., chest pain, shortness of breath, blood in sputum) and displays an immediate medical advisory banner.
  - Flags predictions when confidence is below 40% or when the top two candidates have near-identical probability.
  - Warns users when only a single symptom is selected, advising that more clinical context is required.
- **Local SQLite Logging with Versioning:** Persists timestamped prediction requests in `backend/predictions.db` with model version tags (`v2.0` vs legacy `v1.0`) without deleting prior history.
- **100% Offline & Private:** Operates entirely locally without external AI APIs, LLMs, or paid cloud services.

---

## 2. Technology Stack

| Component | Technologies |
|---|---|
| **Frontend** | React.js 19, Vite, Tailwind CSS v4, Axios, Lucide React |
| **Backend** | Python 3.13+, FastAPI, Uvicorn, Pydantic |
| **Machine Learning** | Scikit-learn, Pandas, NumPy, Joblib |
| **Database** | SQLite (native Python `sqlite3`) with schema versioning |

---

## 3. Project Folder Structure

```text
ML_Disease_Project/
│
├── backend/
│   ├── data/
│   │   └── ml.csv               # Dataset (38,434 rows, 166 cols: 165 symptoms + prognosis)
│   │
│   ├── models/
│   │   ├── best_model.pkl       # Saved Joblib model (Random Forest v2.0)
│   │   ├── label_encoder.pkl    # Disease target label encoder (42 classes)
│   │   ├── feature_columns.pkl  # Exact 165-symptom feature order
│   │   └── model_results.json   # Comprehensive 8-model benchmark results
│   │
│   ├── preprocessing.py         # Group-aware splitting & data audit
│   ├── train_models.py          # 8-model training suite (Bagging, Voting, Stacking, Base)
│   ├── main.py                  # FastAPI server, safety guards & prediction endpoints
│   ├── database.py              # SQLite PredictionHistory CRUD operations with versioning
│   ├── requirements.txt         # Python dependencies
│   └── predictions.db           # SQLite database file
│
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── Navbar.jsx       # Navigation tabs & backend health indicator
│   │   │   └── Footer.jsx       # Disclaimer & project footer
│   │   ├── pages/
│   │   │   ├── Home.jsx         # Project overview, ensemble methods cards & guide
│   │   │   ├── Predict.jsx      # Dynamic symptom selector, safety advisory, differential predictions
│   │   │   ├── Models.jsx       # 8-model comparison table, accuracy charts & ensemble breakdown
│   │   │   └── History.jsx      # SQLite prediction logs table with version tags
│   │   ├── App.jsx              # Tab navigation & layout wrapper
│   │   ├── main.jsx             # React entry point
│   │   └── index.css            # Tailwind CSS styling
│   │
│   ├── package.json
│   ├── vite.config.js           # Vite config with backend proxy
│   └── index.html
│
├── .gitignore
└── README.md
```

---

## 4. Dataset Inspection & Audit (`backend/data/ml.csv`)

Analysis of the preprocessed dataset:

1. **Shape:** 38,434 rows × 166 columns.
2. **Missing Values:** 0 null values across all features and target columns.
3. **Target Variable (`prognosis`):** 42 unique disease classes (ranging from 203 to 1,219 samples per class).
4. **Features:** 165 binary symptom indicators (0 or 1).
5. **Exact Duplicate Full Rows:** 0 duplicate full rows.
6. **Unique Symptom Combinations:** 37,506 unique symptom patterns.
7. **Symptom Patterns with Differential Overlap:** 833 combinations represent shared multi-symptom patterns across diseases, reflecting realistic clinical presentation overlap.
8. **Group-Aware Splitting:**
   - 30,764 training samples (80%)
   - 7,670 held-out test samples (20%)
   - **Pattern overlap between train and test: 0** (strictly zero data leakage).

---

## 5. Model Training & Comparison Results

All 8 algorithms were evaluated on the 30,764 training partition using 3-fold group-aware cross-validation and evaluated on the held-out 7,670 test partition (`random_state=42`):

| Algorithm | Model Type | Train Acc | Group CV Acc | Test Acc | Precision | Recall | F1-Score | Train Time |
|---|---|---|---|---|---|---|---|---|
| **Random Forest** *(Selected)* | Ensemble (Bagging) | **97.53%** | **94.12%** | **94.24%** | **94.28%** | **94.24%** | **94.22%** | 1.29s |
| **Voting Classifier** | Ensemble (Voting) | 97.05% | 94.12% | 94.29% | 94.47% | 94.29% | 94.31% | 8.14s |
| **Stacking Classifier** | Ensemble (Stacking) | 97.08% | 93.94% | 94.33% | 94.43% | 94.33% | 94.34% | 19.88s |
| **Logistic Regression** | Linear Model | 94.75% | 93.96% | 94.24% | 94.40% | 94.24% | 94.25% | 2.02s |
| **SVM (Linear Calibrated)** | Maximum Margin | 94.41% | 93.85% | 94.20% | 94.39% | 94.20% | 94.21% | 4.17s |
| **Naive Bayes (Multinomial)** | Probabilistic | 93.84% | 93.66% | 94.12% | 94.41% | 94.12% | 94.08% | 1.22s |
| **KNN (k=5)** | Instance-Based | 95.12% | 92.77% | 93.48% | 93.54% | 93.48% | 93.49% | 0.02s |
| **Decision Tree** | Single Tree | 97.53% | 92.69% | 93.35% | 93.48% | 93.35% | 93.38% | 1.07s |

### Ensemble Comparison & Final Model Selection

1. **Bagging (Random Forest):** Combines 100 decorrelated decision trees using bootstrap aggregating and random feature subsets. It achieves the top cross-validation accuracy (**94.12%**) with a weighted F1-score of **94.12%**, matches high test accuracy (**94.24%**), and executes in **1.29 seconds**.
2. **Soft Voting Classifier:** Computes an unweighted average of predicted class probabilities across Random Forest, Calibrated Linear SVM, and Logistic Regression. Achieves identical CV accuracy (**94.12%**) and 94.29% test accuracy, but requires **8.14 seconds** (6.3× slower) due to training three models and calibrating SVM.
3. **Stacking Classifier:** Uses out-of-fold predictions from base estimators (Random Forest, Calibrated SVM, Logistic Regression) as meta-features for a Logistic Regression meta-learner. Achieves 93.94% CV accuracy and 94.33% test accuracy, but requires **19.88 seconds** (15.4× slower) and adds architectural complexity.
4. **Final Model Decision:** Per the cross-validation selection criterion, because Voting and Stacking do not meaningfully exceed Random Forest on cross-validation accuracy (94.12%), **Random Forest is retained as the production model**. It offers identical generalization, superior inference latency, simpler explainability, and robust performance without meta-model overhead.

---

## 6. Installation & Execution Guide

### Prerequisites
- Python 3.10+
- Node.js 18+ and npm

### Backend Setup
```bash
cd backend
pip install -r requirements.txt
python train_models.py
python -m uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```
* API Server: `http://127.0.0.1:8000`
* Swagger Interactive Docs: `http://127.0.0.1:8000/docs`

### Frontend Setup (Second Terminal)
```bash
cd frontend
npm install
npm run dev
```
* Web Application: `http://127.0.0.1:5173`

---

## 7. API Endpoints Reference

| Endpoint | Method | Description |
|---|---|---|
| `/api/health` | GET | System status, model loaded status, and model metadata |
| `/api/symptoms` | GET | List of all 165 recognized symptoms in alphabetical order |
| `/api/models` | GET | Comparison metrics for all 8 trained algorithms and ensemble details |
| `/api/predict` | POST | Disease prediction with differential diagnosis, uncertainty flags & safety warnings |
| `/api/history` | GET | Historical prediction logs stored in SQLite database |
| `/api/history` | DELETE | Clears prediction history from SQLite database |
