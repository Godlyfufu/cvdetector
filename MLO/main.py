import pandas as pd
from flask import Flask, render_template, request
from pycaret.classification import *
import mlflow

# Load the saved model
final_model = load_model('final_model')

# Initialize the Flask app
app = Flask(__name__)
with mlflow.start_run(run_id="your_run_id") as run:
    model_uri = f"runs:/{run.info.run_id}/model"
    model = mlflow.sklearn.load_model(model_uri)

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    # Get input values from the form
    age = float(request.form['age'])
    gender = request.form['gender']
    chest_pain = request.form['chest_pain']
    resting_BP = float(request.form['resting_BP'])
    cholesterol = float(request.form['cholesterol'])
    fasting_BS = float(request.form['fasting_BS'])
    resting_ECG = request.form['resting_ECG']
    max_HR = float(request.form['max_HR'])
    exercise_angina = request.form['exercise_angina']
    old_peak = float(request.form['old_peak'])
    ST_slope = request.form['ST_slope']

    # Create a DataFrame with the input data
    input_data = pd.DataFrame({
        'age': [age],
        'gender': [gender],
        'chest_pain': [chest_pain],
        'resting_BP': [resting_BP],
        'cholesterol': [cholesterol],
        'fasting_BS': [fasting_BS],
        'resting_ECG': [resting_ECG],
        'max_HR': [max_HR],
        'exercise_angina': [exercise_angina],
        'old_peak': [old_peak],
        'ST_slope': [ST_slope]
    })

    # Use the preprocessed input data to make predictions with the loaded model
    prediction_result = predict_model(final_model, data=input_data)

    # Get the column name of the prediction result (it might be different from 'Label')
    prediction_column = prediction_result.columns[-1]

    # Extract the actual prediction label
    prediction_label = int(prediction_result.iloc[0][prediction_column])

    # Get the column name of the prediction score (it might be different from 'Score')
    score_column = prediction_result.columns[-1]
    print(score_column)
    # Extract the actual prediction score (probability of positive class)
    prediction_score = float(prediction_result.iloc[0][score_column])
    print(prediction_score)


    if prediction_label == 0:
        prediction_label = 'There are no Cardio Vascular Issues'
    else:
        prediction_label = ' There are Cardio Vascular Issues Present'

    return render_template('index.html', prediction_result=prediction_label, prediction_score=prediction_score)

if __name__ == '__main__':
    app.run(port=8080, debug=True)

