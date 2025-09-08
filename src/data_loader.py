import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
from sklearn.datasets import load_breast_cancer
import os
from src.config import MODEL_DIR

def load_dataset(csv_path=None, test_size=0.2, val_size=0.1, random_state=42):
    if csv_path:
        df = pd.read_csv(csv_path)
        X = df.drop(columns=["label"]).values
        y = df["label"].values
    else:
        dataset = load_breast_cancer()
        X = dataset.data
        y = dataset.target
    
    # Split into train/val/test
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=(test_size+val_size), stratify=y, random_state=random_state
    )
    rel_val = val_size / (test_size+val_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=rel_val, stratify=y_tmp, random_state=random_state
    )

    # Scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # Save scaler for inference
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)
