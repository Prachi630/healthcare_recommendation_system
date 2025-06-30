from preprocessing.clean_data import load_data
from exploration.explore_data import explore_data
from modeling.feature_engineering import preprocess_data
from modeling.model_train import train_model
from modeling.evaluate_model import evaluate_model
from recommendation.generate_recommendation import generate_recommendation

def main():

    data = load_data()
    explore_data(data)

    
    X, y, preprocessor = preprocess_data(data)


    model, X_test, y_test = train_model(X, y, preprocessor)


    evaluate_model(model, X_test, y_test)


    generate_recommendation(model)

if __name__ == "__main__":
    main()
