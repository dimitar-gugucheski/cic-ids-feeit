import pandas as pd
import numpy as np
import requests
import time

# Config
CSV_FILE = "03-02-2018.csv"
TARGET_API_URL = "http://localhost:8031/predict" 
BATCH_SIZE = 1000

def send_bulk_malicious_traffic():
    print(f"Loading {CSV_FILE}")
    try:
        df = pd.read_csv(CSV_FILE, low_memory=False)
    except FileNotFoundError:
        print("File not found")
        return

    # Clean
    df.columns = df.columns.str.strip()
    if 'Label' not in df.columns:
        print("Missing Label column")
        return
    df['Label'] = df['Label'].astype(str).str.strip()

    # Filter
    malicious_df = df[df['Label'].str.lower() != 'benign']
    total_malicious = len(malicious_df)

    if total_malicious == 0:
        print("No malicious flows")
        return
        
    print(f"Found {total_malicious:,} flows")

    # Format
    features_df = malicious_df.drop(columns=['Label'])
    features_df = features_df.select_dtypes(include=[np.number])
    features_df.replace([np.inf, -np.inf], 0.0, inplace=True)
    features_df.fillna(0.0, inplace=True)
    records = features_df.to_dict(orient='records')

    total_caught = 0
    total_missed = 0

    print(f"Sending batches of {BATCH_SIZE}")
    print("-" * 50)

    # Batch
    for i in range(0, total_malicious, BATCH_SIZE):
        batch_records = records[i:i + BATCH_SIZE]
        payload = {"flows": [{"features": rec} for rec in batch_records]}

        # Send
        try:
            response = requests.post(TARGET_API_URL, json=payload)
            
            if response.status_code == 422:
                print("Format error:")
                print(response.text)
                return
                
            response.raise_for_status() 
            
            # Tally
            results = response.json().get("results", [])
            for res in results:
                if res['is_malicious']:
                    total_caught += 1
                else:
                    total_missed += 1
            
            print(f"Sent {min(i + BATCH_SIZE, total_malicious):,}/{total_malicious:,}")

        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}")
            break

        # Delay
        time.sleep(0.1) 

    # Report
    print("-" * 50)
    print("Done")
    print("-" * 50)
    print(f"Sent: {total_malicious:,}")
    print(f"Caught: {total_caught:,}")
    print(f"Missed: {total_missed:,}")
    
    if total_malicious > 0:
        print(f"Accuracy: {(total_caught / total_malicious) * 100:.2f}%")
    print("-" * 50)

if __name__ == "__main__":
    send_bulk_malicious_traffic()