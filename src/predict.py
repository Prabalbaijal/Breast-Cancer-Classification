import numpy as np
import pickle
from tensorflow.keras.models import load_model
from .config import MODEL_PATH, SCALER_PATH
from .data_loader import load_data

def predict_single(input_data):
    # Load scaler
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)

    # Load trained model
    model = load_model(MODEL_PATH)

    # Preprocess input
    input_data_as_numpy_array = np.asarray(input_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    input_data_std = scaler.transform(input_data_reshaped)

    # Predict
    prediction = model.predict(input_data_std, verbose=0)
    prediction_label = np.argmax(prediction)

    return prediction_label


if __name__ == "__main__":
    # Get data
    X_train, X_test, y_train, y_test = load_data()
    
    # Pick one sample
    sample = X_test[0]
    true_label = y_test[0]   # notebook style, no .iloc

    pred = predict_single(sample)

    print("True Label:", "Benign (1)" if true_label == 1 else "Malignant (0)")
    print("Predicted Label:", "Benign (1)" if pred == 1 else "Malignant (0)")
