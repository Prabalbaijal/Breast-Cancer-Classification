import pandas as pd
import sklearn.datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
from src.config import SCALER_PATH

def load_data():
    # Load dataset
    breast_cancer_dataset = sklearn.datasets.load_breast_cancer()

    # Create dataframe with features
    data_frame = pd.DataFrame(breast_cancer_dataset.data, columns=breast_cancer_dataset.feature_names)

    # Add target column (same as notebook)
    data_frame['label'] = breast_cancer_dataset.target

    # Separate features and target
    X = data_frame.drop(columns='label')
    y = data_frame['label']

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

    # Scale
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    joblib.dump(scaler, SCALER_PATH)

    return X_train, X_test, y_train, y_test
