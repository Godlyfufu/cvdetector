import pandas as pd
from pycaret.classification import *

# Load the saved model
final_model = load_model('final_model_medical2')


# Define fixed input values for prediction
input_data = pd.DataFrame({
    'age': [49],
    'gender': ['F'],
    'chest_pain': ['NAP'],
    'resting_BP': [160],
    'cholesterol': [180],
    'fasting_BS': [0],
    'resting_ECG': ['Normal'],
    'max_HR': [156],
    'exercise_angina': ['N'],
    'old_peak': [1.0],
    'ST_slope': ['Flat']
})

# Use the preprocessed input data to make predictions with the loaded model
prediction_result = predict_model(final_model, data=input_data)

# Get the column name of the prediction result (it might be different from 'Label')
prediction_column = prediction_result.columns[-1]

# Extract the actual prediction class label (0 or 1)
prediction_label = int(prediction_result.iloc[0][prediction_column])

print(f"Predicted class: {prediction_label}")

# Get the column name of the prediction score (it might be different from 'Score')
score_column = prediction_result.columns[-1]

# Extract the actual prediction score (probability of positive class)
prediction_score = float(prediction_result.iloc[0][score_column])

print(f"Predicted score: {prediction_score}")


