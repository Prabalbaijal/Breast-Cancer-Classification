import numpy as np
from tensorflow.keras.models import load_model
import joblib

# Paths to model and scaler
MODEL_PATH = "models/model.h5"
SCALER_PATH = "models/scaler.pkl"

# Feature names (from breast cancer dataset)
FEATURE_NAMES = [
    'mean radius', 'mean texture', 'mean perimeter', 'mean area', 'mean smoothness',
    'mean compactness', 'mean concavity', 'mean concave points', 'mean symmetry', 'mean fractal dimension',
    'radius error', 'texture error', 'perimeter error', 'area error', 'smoothness error',
    'compactness error', 'concavity error', 'concave points error', 'symmetry error', 'fractal dimension error',
    'worst radius', 'worst texture', 'worst perimeter', 'worst area', 'worst smoothness',
    'worst compactness', 'worst concavity', 'worst concave points', 'worst symmetry', 'worst fractal dimension'
]

def predict_single(input_data):
    # Load scaler
    scaler = joblib.load(SCALER_PATH)

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
    print("Choose input mode:")
    print("1. Step-by-step feature input (recommended)")
    print("2. Single-line copy-paste (30 values space-separated)")
    mode = input("Enter 1 or 2: ").strip()

    user_input_list = []

    if mode == "1":
        print("\nEnter values for the following features:")
        for feature in FEATURE_NAMES:
            while True:
                try:
                    value = float(input(f"{feature}: "))
                    user_input_list.append(value)
                    break
                except ValueError:
                    print("Invalid input! Please enter a numeric value.")
    elif mode == "2":
        while True:
            user_input = input("\nEnter 30 values space-separated (same order as features):\n")
            try:
                user_input_list = [float(x) for x in user_input.strip().split()]
                if len(user_input_list) != 30:
                    raise ValueError("Expected exactly 30 values.")
                break
            except ValueError as e:
                print("Invalid input:", e)
    else:
        print("Invalid choice!")
        exit()

    # Predict
    pred = predict_single(user_input_list)

    # Output
    print("\nPrediction Result:")
    print("Predicted Label:", "Benign (1)" if pred == 1 else "Malignant (0)")
