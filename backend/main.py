

import os
import json
import joblib
import numpy as np
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from database import init_db, save_prediction, get_history

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
BEST_MODEL_PATH = os.path.join(MODELS_DIR, "best_model.pkl")
LABEL_ENCODER_PATH = os.path.join(MODELS_DIR, "label_encoder.pkl")
FEATURE_COLUMNS_PATH = os.path.join(MODELS_DIR, "feature_columns.pkl")
RESULTS_JSON_PATH = os.path.join(MODELS_DIR, "model_results.json")

# Red-flag urgent symptoms that warrant immediate clinical emergency guidance
URGENT_RED_FLAG_SYMPTOMS = {
    "sharp chest pain",
    "shortness of breath",
    "difficulty breathing",
    "hurts to breath",
    "arm weakness",
    "leg weakness",
    "focal weakness",
    "abnormal involuntary movements",
    "blood in urine",
    "depressive or psychotic symptoms"
}

# Initialize database
init_db()

# Load saved ML artifacts at startup
best_model = None
label_encoder = None
feature_columns: List[str] = []
model_metadata: Dict[str, Any] = {}


def reload_ml_artifacts():
    global best_model, label_encoder, feature_columns, model_metadata
    try:
        if os.path.exists(BEST_MODEL_PATH):
            best_model = joblib.load(BEST_MODEL_PATH)
        if os.path.exists(LABEL_ENCODER_PATH):
            label_encoder = joblib.load(LABEL_ENCODER_PATH)
        if os.path.exists(FEATURE_COLUMNS_PATH):
            feature_columns = joblib.load(FEATURE_COLUMNS_PATH)
        if os.path.exists(RESULTS_JSON_PATH):
            with open(RESULTS_JSON_PATH, "r", encoding="utf-8") as f:
                model_metadata = json.load(f)
    except Exception as e:
        print(f"Warning: Failed to load models: {e}")


reload_ml_artifacts()
best_model_name = model_metadata.get("best_model_name", "Random Forest")

# Initialize FastAPI app
app = FastAPI(
    title="AI-Based Disease Prediction System",
    description="A multi-class disease prediction API powered by scikit-learn",
    version="2.1.0"
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic Schemas
class PredictionRequest(BaseModel):
    symptoms: List[str] = Field(default_factory=list, description="List of symptom names selected by user")


class TopPrediction(BaseModel):
    disease: str
    score: float


class PredictionResponse(BaseModel):
    predicted_disease: str
    model_name: str
    top_predictions: List[TopPrediction]
    disclaimer: str
    is_uncertain: bool = False
    uncertainty_note: Optional[str] = None
    insufficient_info: bool = False
    insufficient_info_note: Optional[str] = None
    urgent_guidance: Optional[str] = None


def format_symptom_label(symptom_name: str) -> str:
    """Formats column names like 'shortness of breath' or 'joint_pain' into readable labels."""
    clean = symptom_name.replace(".1", "").replace("_", " ").strip()
    return clean.title()


# Endpoints

@app.get("/api/health", tags=["Health"])
def get_health():
    """
    Check whether backend and saved ML model are available.
    """
    if not best_model or not label_encoder:
        reload_ml_artifacts()

    is_ready = bool(best_model and label_encoder and len(feature_columns) > 0)
    current_best = model_metadata.get("best_model_name", "Random Forest")
    return {
        "status": "healthy" if is_ready else "degraded",
        "model_loaded": is_ready,
        "best_model_name": current_best,
        "dataset_version": model_metadata.get("dataset_version", "v2.0"),
        "total_features": len(feature_columns),
        "total_diseases": len(label_encoder.classes_) if label_encoder else 0
    }


@app.get("/api/symptoms", tags=["Symptoms"])
def get_symptoms():
    """
    Return all available symptoms from the dataset in the exact feature order.
    """
    if not feature_columns:
        reload_ml_artifacts()

    if not feature_columns:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Feature columns not loaded. Please train models first."
        )

    symptoms_data = [
        {
            "id": col,
            "label": format_symptom_label(col)
        }
        for col in feature_columns
    ]
    return {
        "symptoms": symptoms_data,
        "total": len(symptoms_data)
    }


@app.post("/api/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict_disease(request: PredictionRequest):
   
    if not best_model or not label_encoder or not feature_columns:
        reload_ml_artifacts()

    if not best_model or not label_encoder or not feature_columns:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Model is not ready. Ensure train_models.py has been run."
        )

    selected_symptoms = request.symptoms
    # 1. Handle empty symptom selections
    if not selected_symptoms or len(selected_symptoms) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No symptoms selected. Please select at least one symptom to evaluate."
        )

    # 2. Validate symptoms against current feature columns
    valid_features_set = set(feature_columns)
    invalid_symptoms = [s for s in selected_symptoms if s not in valid_features_set]
    if invalid_symptoms:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unrecognized symptoms provided: {invalid_symptoms}"
        )

    # 3. Build binary vector in exact training feature order
    feature_vector = np.array([
        1 if feat in selected_symptoms else 0 for feat in feature_columns
    ]).reshape(1, -1)

    # 4. Predict most likely disease
    predicted_idx = best_model.predict(feature_vector)[0]
    predicted_disease = label_encoder.inverse_transform([predicted_idx])[0]

    # 5. Top 3 predictions if probabilities supported
    top_predictions: List[Dict[str, Any]] = []
    if hasattr(best_model, "predict_proba"):
        probabilities = best_model.predict_proba(feature_vector)[0]
        # Sort probabilities descending
        top_indices = np.argsort(probabilities)[::-1][:3]
        for idx in top_indices:
            score = float(probabilities[idx])
            if score > 0.001:  # Filter negligible artifacts
                disease_name = label_encoder.inverse_transform([idx])[0]
                top_predictions.append({
                    "disease": disease_name,
                    "score": round(score, 4)
                })

    if not top_predictions:
        top_predictions = [{"disease": predicted_disease, "score": 1.0}]

    current_best = model_metadata.get("best_model_name", "Random Forest")

    # 6. Evaluate uncertainty & insufficient symptom information
    is_uncertain = False
    uncertainty_note = None
    insufficient_info = False
    insufficient_info_note = None

    if len(selected_symptoms) == 1:
        insufficient_info = True
        insufficient_info_note = (
            "Only a single symptom was provided. Single non-specific symptoms rarely identify "
            "a unique disease condition. Consider selecting additional accompanying symptoms for better accuracy."
        )

    top_score = top_predictions[0]["score"]
    second_score = top_predictions[1]["score"] if len(top_predictions) > 1 else 0.0
    margin = top_score - second_score

    # Flag uncertainty if top score is low (< 0.40) or close competition with 2nd candidate (margin < 0.08)
    if top_score < 0.40 or (len(top_predictions) > 1 and margin < 0.08) or (len(selected_symptoms) == 1 and top_score < 0.50):
        is_uncertain = True
        uncertainty_note = (
            f"High Diagnostic Uncertainty: The statistical likelihood for the top condition is only {round(top_score * 100, 1)}%. "
            f"The reported symptom profile overlaps significantly with other potential conditions "
            f"({top_predictions[0]['disease']} vs {second_score*100:.1f}% for {top_predictions[1]['disease'] if len(top_predictions) > 1 else 'others'}). "
            "This result should NOT be interpreted as a confident diagnosis."
        )

    # 7. Safety guidance for potentially urgent / red-flag symptoms
    urgent_guidance = None
    present_urgent_symptoms = [s for s in selected_symptoms if s in URGENT_RED_FLAG_SYMPTOMS]
    if present_urgent_symptoms:
        urgent_labels = [format_symptom_label(s) for s in present_urgent_symptoms]
        urgent_guidance = (
            f"Urgent Safety Advisory: You have selected symptoms that may indicate an urgent medical situation "
            f"({', '.join(urgent_labels)}). If you or the patient are experiencing sudden, severe, or worsening "
            "distress, severe chest pressure, or difficulty breathing, please seek immediate emergency medical care "
            "(call local emergency services or visit the nearest emergency room). Do not rely on an AI demonstration tool."
        )

    # 8. Save prediction to SQLite database with model_version="v2.0"
    try:
        save_prediction(
            symptoms=selected_symptoms,
            predicted_disease=predicted_disease,
            model_name=current_best,
            model_version="v2.0"
        )
    except Exception as db_err:
        print(f"Warning: Failed to save prediction to history: {db_err}")

    # 9. Standard medical disclaimer
    disclaimer = (
        "Educational Demonstration Notice: This prediction is generated solely by a statistical "
        "machine learning model (Random Forest) for college academic demonstration purposes. It is NOT a clinical "
        "diagnosis, medical triage decision, or prescription. No medications are prescribed. Always consult a licensed "
        "physician for any health symptoms."
    )

    return {
        "predicted_disease": predicted_disease,
        "model_name": current_best,
        "top_predictions": top_predictions,
        "disclaimer": disclaimer,
        "is_uncertain": is_uncertain,
        "uncertainty_note": uncertainty_note,
        "insufficient_info": insufficient_info,
        "insufficient_info_note": insufficient_info_note,
        "urgent_guidance": urgent_guidance
    }


@app.get("/api/models", tags=["Model Benchmark"])
def get_model_benchmarks():
    """
    Return model comparison results and dataset analysis from model_results.json.
    """
    if os.path.exists(RESULTS_JSON_PATH):
        with open(RESULTS_JSON_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    if model_metadata:
        return model_metadata
    raise HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail="Model results not found. Please run train_models.py first."
    )


@app.get("/api/history", tags=["History"])
def get_prediction_history(limit: int = 50):
    """
    Return recent prediction history records from SQLite database with versioning.
    """
    try:
        history = get_history(limit=limit)
        return {"history": history, "total": len(history)}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch history: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
