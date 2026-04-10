import json
import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict

app = FastAPI(title="IDS XGBoost API", version="1.0")

# Globals
model = None
expected_columns = []

@app.on_event("startup")
def load_artifacts():
    global model, expected_columns
    
    # Columns
    with open("model_columns.json", "r") as f:
        expected_columns = json.load(f)
        
    # Model
    model = joblib.load("xgboost_model.joblib")

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
                
        # Inference
        predictions = model.predict(df_aligned)
        probabilities = model.predict_proba(df_aligned)
        
        # Formatting
        results = []
        for i, pred in enumerate(predictions):
            # Mapping
            label = "Benign" if pred == 0 else "Malicious"
            confidence = float(probabilities[i][pred])
            
            results.append({
                "prediction": label,
                "confidence": confidence,
                "is_malicious": bool(pred == 1)
            })
            
        return {"results": results}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    return {"status": "healthy"}