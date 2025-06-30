import pandas as pd

def load_data(path='data/healthcare_data.csv'):
    data = pd.read_csv(path)
    return data
