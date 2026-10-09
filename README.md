# AI-Based Disease Prediction System

A complete, beginner-friendly college project that predicts possible diseases based on user-selected symptoms using supervised multiclass machine learning. The system is built with a **React (Vite + Tailwind CSS)** frontend, a **FastAPI (Python)** REST backend, a locally trained **Scikit-learn** classification engine, and an **SQLite** database for storing prediction history.

> **Medical Disclaimer:** This system is developed strictly for educational and academic demonstration purposes. It does not provide clinical medical diagnoses, prescription recommendations, or treatment plans. Always consult a licensed healthcare professional for medical concerns.

---

## 1. System Architecture & Features

- **Dynamic Symptom Selection:** 165 standardized symptoms loaded dynamically from the new dataset (`backend/data/ml.csv`) with real-time search filtering and interactive tag chips.
- **Zero-Leakage Group-Aware Splitting:** Splitting performed across unique symptom combinations (groups) ensuring **0% pattern overlap** between training and testing sets.
- **Comparative ML Benchmarking:** 6 classification algorithms evaluated across Training Accuracy, Group-Aware Cross-Validation Accuracy, and Held-Out Test Accuracy:
  - Decision Tree Classifier
  - Random Forest Classifier (Selected Best)
  - K-Nearest Neighbors (KNN)
  - Support Vector Machine (Linear SVM with probability calibration)
  - Logistic Regression
  - Naive Bayes (Multinomial)
- **Differential Disease Ranking:** Outputs the primary predicted disease along with top 3 differential candidates and relative statistical probability scores.
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
│   │   └── ml.csv               # Current preprocessed dataset (38,434 rows, 166 cols)
│   │
│   ├── models/
│   │   ├── best_model.pkl       # Saved Joblib model (Random Forest v2.0)
│   │   ├── label_encoder.pkl    # Disease target label encoder (42 classes)
│   │   ├── feature_columns.pkl  # Exact 165-symptom feature order
│   │   └── model_results.json   # Model performance metrics & benchmarks
│   │
│   ├── preprocessing.py         # Data inspection & group-aware splitting
│   ├── train_models.py          # 6-model training, GroupKFold CV, & artifact saving
│   ├── main.py                  # FastAPI server & prediction endpoints
│   ├── database.py              # SQLite PredictionHistory CRUD operations
│   ├── requirements.txt         # Python dependencies
│   └── predictions.db           # SQLite database file
│
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── Navbar.jsx       # Navigation tabs & backend health indicator
│   │   │   └── Footer.jsx       # Disclaimer & project footer
│   │   ├── pages/
│   │   │   ├── Home.jsx         # Overview & how-the-system-works cards
│   │   │   ├── Predict.jsx      # Symptom selector & differential prediction
│   │   │   ├── Models.jsx       # Benchmark comparison table & charts
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

Analysis of the new preprocessed dataset:

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

All 6 algorithms were trained on the 30,764 training partition using 3-fold group-aware cross-validation and evaluated on the held-out 7,670 test partition (`random_state=42`):

| Model | Training Accuracy | Group CV Accuracy | Testing Accuracy | Precision (Weighted) | Recall (Weighted) | F1-Score (Weighted) |
|---|---|---|---|---|---|---|
| **Random Forest** *(Selected)* | **97.53%** | **94.12%** | **94.24%** | **94.28%** | **94.24%** | **94.22%** |
| **Logistic Regression** | 94.75% | 93.96% | 94.24% | 94.40% | 94.24% | 94.25% |
| **SVM (Linear Calibrated)** | 94.41% | 93.85% | 94.20% | 94.39% | 94.20% | 94.21% |
| **Naive Bayes (Multinomial)** | 93.84% | 93.66% | 94.12% | 94.41% | 94.12% | 94.08% |
| **KNN (k=5)** | 95.12% | 92.77% | 93.48% | 93.54% | 93.48% | 93.49% |
| **Decision Tree** | 97.53% | 92.69% | 93.35% | 93.48% | 93.35% | 93.38% |

### Best Model Selection
**Random Forest Classifier** was selected based on training-set group-aware cross-validation:
- Highest mean cross-validation accuracy (**94.12%**)
- Highest weighted cross-validation F1-score (**94.12%**)
- Held-out test accuracy of **94.24%** and F1-score of **94.22%**
- Native ensemble probability estimation (`predict_proba`)

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
* Swagger Docs: `http://127.0.0.1:8000/docs`

### Frontend Setup (Second Terminal)
```bash
cd frontend
npm install
npm run dev
```
* Web Application: `http://127.0.0.1:5173`
