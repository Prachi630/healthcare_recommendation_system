from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def preprocess_data(data):
    
    X = data.drop('Class', axis=1)
    y = data['Class']

    
    numerical_features = ['Recency', 'Frequency', 'Monetary', 'Time']
    categorical_features = [] 

    numerical_pipeline = Pipeline([
        ('scaler', StandardScaler())
    ])

    preprocessor = ColumnTransformer([
        ('num', numerical_pipeline, numerical_features)
    ])

    return X, y, preprocessor
