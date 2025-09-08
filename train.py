from src.data_loader import load_dataset
from src.model import build_model
from src.config import MODEL_DIR
import os
import tensorflow as tf

def train_model(csv_path=None, epochs=30, batch_size=32):
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_dataset(csv_path)

    model = build_model(X_train.shape[1])

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join(MODEL_DIR, "model.h5"),
            monitor="val_auc",
            save_best_only=True,
            mode="max"
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_auc", patience=5, mode="max", restore_best_weights=True
        )
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        class_weight={0:1, 1:2}  # handle imbalance
    )

    print("Test evaluation:", model.evaluate(X_test, y_test))
    return model, history

if __name__ == "__main__":
    train_model()
