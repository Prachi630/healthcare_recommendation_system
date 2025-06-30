import pandas as pd

def generate_recommendation(model):
 
    example_patient = pd.DataFrame({
        'Recency': [10],
        'Frequency': [4],
        'Monetary': [300],
        'Time': [5]
    })

   
    pred = model.predict(example_patient)


    recommendation_mapping = {
        0: 'No action needed',
        1: 'Regular check-up',
        2: 'Lifestyle changes',
        3: 'Medication'
    }

    print(f"\n💡 Personalized Recommendation: {recommendation_mapping.get(pred[0], 'Unknown')}")
