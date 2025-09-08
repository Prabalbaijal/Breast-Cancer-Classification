# Data loading and preprocessing

import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle
from .config import SCALER_PATH

def load_data():
    # Load sklearn dataset
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=2, stratify=y
    )

    # Standardize
    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(X_train)
    X_test_std = scaler.transform(X_test)

    # Save scaler
    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)

    return X_train_std, X_test_std, y_train, y_test
