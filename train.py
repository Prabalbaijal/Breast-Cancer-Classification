# Training script

from src.data_loader import load_data
from src.model import create_model
from src.config import MODEL_PATH
from src.train_utils import plot_history

def train():
    X_train, X_test, y_train, y_test = load_data()

    model = create_model(input_dim=X_train.shape[1])

    history = model.fit(
        X_train, y_train,
        validation_split=0.1,
        epochs=10,
        verbose=1
    )

    plot_history(history)

    loss, accuracy = model.evaluate(X_test, y_test)
    print(f" Test Accuracy: {accuracy*100:.2f}%")

    model.save(MODEL_PATH)
    print(f" Model saved at {MODEL_PATH}")

if __name__ == "__main__":
    train()
