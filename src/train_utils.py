import matplotlib.pyplot as plt
import os

PLOTS_DIR = "plots"
os.makedirs(PLOTS_DIR, exist_ok=True)  # Create folder if not exists

def plot_history(history):
    # Accuracy
    plt.figure()
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Training data', 'Validation data'], loc='lower right')
    plt.savefig(os.path.join(PLOTS_DIR, "accuracy.png"))
    plt.close()  # Close figure to free memory

    # Loss
    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Training data', 'Validation data'], loc='upper right')
    plt.savefig(os.path.join(PLOTS_DIR, "loss.png"))
    plt.close()
