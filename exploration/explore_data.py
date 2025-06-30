import matplotlib.pyplot as plt
import seaborn as sns

def explore_data(data):
    print("📊 Head of the Data:\n", data.head())
    print("\n📋 Description:\n", data.describe())
    print("\n🔍 Missing Values:\n", data.isnull().sum())


    plt.figure(figsize=(10, 6))
    sns.heatmap(data.corr(numeric_only=True), annot=True, cmap='coolwarm')
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.show()
