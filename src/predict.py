import numpy as np
import pickle
import os
from tensorflow.keras.models import load_model
from .config import MODEL_PATH, SCALER_PATH
from .data_loader import load_data
import pandas as pd

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

def predict_single(input_data):
    # Load scaler
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)

    # Load trained model
    model = load_model(MODEL_PATH)

    input_data_as_numpy_array = np.asarray(input_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    input_data_std = scaler.transform(input_data_reshaped)

    prediction = model.predict(input_data_std, verbose=0)
    prediction_label = np.argmax(prediction)

    # Save result
    result_file = os.path.join(RESULTS_DIR, "prediction_results.csv")
    df = pd.DataFrame([{
        "input": input_data.tolist(),
        "prediction": int(prediction_label)
    }])
    if os.path.exists(result_file):
        df.to_csv(result_file, mode='a', header=False, index=False)
    else:
        df.to_csv(result_file, index=False)

    return prediction_label

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_data()
    sample = X_test[0]
    true_label = y_test.iloc[0]

    pred = predict_single(sample)

    print(" True Label:", "Benign (1)" if true_label == 1 else "Malignant (0)")
    print(" Predicted Label:", "Benign (1)" if pred == 1 else "Malignant (0)")
