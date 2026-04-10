import json
import joblib
import pandas as pd
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict

app = FastAPI(title="IDS Autoencoder API", version="1.0")

# Globals
model = None
scaler = None
label_encoder = None
expected_columns = []

@app.on_event("startup")
def load_artifacts():
    global model, scaler, label_encoder, expected_columns
    
    # Columns
    with open("model_columns.json", "r") as f:
        expected_columns = json.load(f)
        
    # Preprocessing
    scaler = joblib.load("scaler.joblib")
    label_encoder = joblib.load("label_encoder.joblib")
    
    # Model
    model = tf.keras.models.load_model("supervised_autoencoder.keras")

# Schemas
class FlowData(BaseModel):
    features: Dict[str, float]

class BatchRequest(BaseModel):
    flows: List[FlowData]

# Endpoint
@app.post("/predict")
def predict(request: BatchRequest):
    try:
        # Conversion
        input_data = [item.features for item in request.flows]
        df = pd.DataFrame(input_data)
        
        # Alignment
        df_aligned = pd.DataFrame()
        for col in expected_columns:
            if col in df.columns:
                df_aligned[col] = df[col]
            else:
                df_aligned[col] = 0.0
                
        # Scaling
        X = df_aligned.values
        X_scaled = scaler.transform(X)
        
        # Inference
        model_outputs = model.predict(X_scaled, verbose=0)
        # Autoencoder returns [reconstruction, classification]
        classification_preds = model_outputs[1] 
        
        # Decoding
        raw_scores = classification_preds.flatten()
        predicted_indices = (raw_scores > 0.5).astype(int)
        confidence_scores = np.where(raw_scores > 0.5, raw_scores, 1.0 - raw_scores)
            
        predicted_labels = label_encoder.inverse_transform(predicted_indices)
        
        # Formatting
        results = []
        for i, label in enumerate(predicted_labels):
            results.append({
                "prediction": label,
                "confidence": float(confidence_scores[i]),
                "is_malicious": str(label).strip().lower() != "benign"
            })
            
        return {"results": results}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    return {"status": "healthy"}