import pandas as pd
import numpy as np
import requests
import time

CSV_FILE = "02-22-2018.csv"  # dataset
BATCH_SIZE = 512 # size
API_URL = "http://localhost:8022/predict" # endpoint
MODEL_NAME = "Model 02-22" # identifier

def run_single_evaluation():
    print(f"Loading dataset from {CSV_FILE} ...")
    try:
        df = pd.read_csv(CSV_FILE, low_memory=False)
    except FileNotFoundError:
        print(f"Error: Cannot find {CSV_FILE}.")
        return

    # clean
    print("Cleaning data...")
    df.columns = df.columns.str.strip() # strip
    df.replace([np.inf, -np.inf], np.nan, inplace=True) # replace
    
    # numeric
    df = df.select_dtypes(include=[np.number]) 
    
    df.fillna(0, inplace=True) # fill
    df = df.astype(float) # cast
    
    records = df.to_dict(orient="records") # convert
    total_flows = len(records) # count
    total_batches = (total_flows // BATCH_SIZE) + (1 if total_flows % BATCH_SIZE != 0 else 0) # calculate
    
    print(f"OK! Sending {total_flows} total flows ({total_batches} batches of {BATCH_SIZE}).\n")

    malicious_count = 0
    benign_count = 0
    error_count = 0

    print(f"--- Beginning deep scan with {MODEL_NAME} ---")
    
    # chunking
    for i in range(0, total_flows, BATCH_SIZE):
        batch = records[i : i + BATCH_SIZE]
        payload = {"flows": [{"features": row} for row in batch]}
        batch_num = (i // BATCH_SIZE) + 1
        
        try:
            response = requests.post(API_URL, json=payload)
            
            if response.status_code == 200:
                results = response.json().get("results", [])
                caught = sum(1 for r in results if r["is_malicious"])
                benign = len(results) - caught
                
                malicious_count += caught
                benign_count += benign
            else:
                error_count += 1
                
        except requests.exceptions.ConnectionError:
            print(f"\nConnection failed for {MODEL_NAME}! port-forwarding on 8022!")
            error_count += 1
            break 
            
        # update
        if batch_num % 100 == 0 or batch_num == total_batches:
            progress_pct = (batch_num / total_batches) * 100
            print(f"   [Batch {batch_num}/{total_batches}] - {progress_pct:.1f}% complete...")

    print(f"End of processing {total_flows} flows for {MODEL_NAME}\n")

    # summary
    print("="*50)
    print(f"SINGLE MODEL EVALUATION REPORT")
    print("="*50)
    print(f"Dataset Scanned    : {CSV_FILE}")
    print(f"Model Tested       : {MODEL_NAME}")
    print(f"Malicious Detected : {malicious_count}")
    print(f"Benign Detected    : {benign_count}")
    print(f"Errors             : {error_count}")
    print("="*50)

if __name__ == "__main__":
    # execute
    run_single_evaluation()