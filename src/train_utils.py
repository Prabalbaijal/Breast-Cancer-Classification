import matplotlib.pyplot as plt
import os

PLOTS_DIR = "plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

def plot_history(history):
    # Accuracy plot
    plt.figure(figsize=(8, 6))
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Model Accuracy')
    plt.legend()
    acc_path = os.path.join(PLOTS_DIR, "accuracy.png")
    plt.savefig(acc_path)
    plt.close()

    # Loss plot
    plt.figure(figsize=(8, 6))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Model Loss')
    plt.legend()
    loss_path = os.path.join(PLOTS_DIR, "loss.png")
    plt.savefig(loss_path)
    plt.close()

    print(f"Training plots saved to {PLOTS_DIR}")
