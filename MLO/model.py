from pycaret.classification import *
import pandas as pd

medical = pd.read_csv('Dataset/02_medical_records.csv')

# Initialize the setup for PyCaret
exp = setup(data=medical, target='cv_issue', session_id=42,
            normalize=True,
            feature_selection=True,
            fix_imbalance=True,
            fix_imbalance_method='smote',
            # bin_numeric_features=['age', 'resting_BP', 'cholesterol', 'max_HR'],
            bin_numeric_features=['age'],
            categorical_features=['gender', 'chest_pain', 'resting_ECG', 'exercise_angina', 'ST_slope'],
            fold=10)

# Compare all models and select the best one based on Accuracy
best_model = compare_models(sort='Accuracy')

# Tune hyperparameters of the selected model using Random Grid Search
tuned_model = tune_model(best_model, n_iter=50, optimize='Accuracy')

# Finalize the tuned model
final_model = finalize_model(tuned_model)

# Save the finalized model
save_model(final_model, 'final_model_medical2')

