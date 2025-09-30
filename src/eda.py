import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer

def run_eda():
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target

    print("🔹 Shape:", df.shape)
    print(df.head())
    print(df['target'].value_counts())

    # Class distribution
    sns.countplot(x="target", data=df)
    plt.title("Target Distribution (0 = Malignant, 1 = Benign)")
    plt.show()

    # Correlation heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(df.corr(), cmap="coolwarm")
    plt.title("Feature Correlation Heatmap")
    plt.show()

if __name__ == "__main__":
    run_eda()
