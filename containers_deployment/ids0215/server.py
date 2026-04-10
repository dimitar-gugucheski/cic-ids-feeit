import os
import json
import joblib
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel 
from typing import List, Dict, Any

# layers

@tf.keras.utils.register_keras_serializable()
class FeatureTokenizer(layers.Layer):
    def __init__(self, num_features, d_token, **kwargs):
        super(FeatureTokenizer, self).__init__(**kwargs)
        self.num_features = num_features
        self.d_token = d_token

    def build(self, input_shape):
        self.feature_weights = self.add_weight(
            shape=(self.num_features, self.d_token),
            initializer="glorot_uniform",
            trainable=True,
        )
        self.feature_biases = self.add_weight(
            shape=(self.num_features, self.d_token),
            initializer="zeros",
            trainable=True,
        )
        self.cls_token = self.add_weight(
            shape=(1, 1, self.d_token),
            initializer="glorot_uniform",
            trainable=True,
        )
        super(FeatureTokenizer, self).build(input_shape)

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        x_expanded = tf.expand_dims(inputs, -1)
        tokens = x_expanded * self.feature_weights + self.feature_biases
        cls_tokens = tf.tile(self.cls_token, [batch_size, 1, 1])
        return tf.concat([cls_tokens, tokens], axis=1)

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_features": self.num_features,
            "d_token": self.d_token
        })
        return config


@tf.keras.utils.register_keras_serializable()
class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, dropout_rate=0.1, **kwargs):
        super(TransformerBlock, self).__init__(**kwargs)
        # attributes
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate

        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential([
            layers.Dense(int(embed_dim * 1.33), activation="relu"),
            layers.Dense(embed_dim),
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(dropout_rate)
        self.dropout2 = layers.Dropout(dropout_rate)

    def call(self, inputs, training=False):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "dropout_rate": self.dropout_rate
        })
        return config

# artifacts

app = FastAPI(title="IDS Inference API", version="1.0")

# globals
model = None
scaler = None
label_encoder = None
expected_columns = []

@app.on_event("startup")
def load_artifacts():
    global model, scaler, label_encoder, expected_columns
    
    # columns
    with open("model_columns.json", "r") as f:
        expected_columns = json.load(f)
        
    # scikit
    scaler = joblib.load("scaler.joblib")
    label_encoder = joblib.load("label_encoder.joblib")
    
    # keras
    model = tf.keras.models.load_model("ids_model.keras", custom_objects={
        "FeatureTokenizer": FeatureTokenizer,
        "TransformerBlock": TransformerBlock
    })
    
    print("All artifacts loaded successfully!")

# schema

class FlowData(BaseModel):
    # flexible
    features: Dict[str, float]

class BatchRequest(BaseModel):
    flows: List[FlowData]

# endpoint

@app.post("/predict")
def predict(request: BatchRequest):
    try:
        # conversion
        input_data = [item.features for item in request.flows]
        df = pd.DataFrame(input_data)
        
        # alignment
        df_aligned = pd.DataFrame()
        for col in expected_columns:
            if col in df.columns:
                df_aligned[col] = df[col]
            else:
                df_aligned[col] = 0.0 # padding
                
        # scaling
        X = df_aligned.values
        X_scaled = scaler.transform(X)
        
        # inference
        predictions = model.predict(X_scaled, verbose=0)
        
        # decoding
        predicted_indices = np.argmax(predictions, axis=1)
        predicted_labels = label_encoder.inverse_transform(predicted_indices)
        confidence_scores = np.max(predictions, axis=1)
        
        # formatting
        results = []
        for i, label in enumerate(predicted_labels):
            results.append({
                "prediction": label,
                "confidence": float(confidence_scores[i]),
                "is_malicious": label != "Benign"
            })
            
        return {"results": results}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": model is not None}