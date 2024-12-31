import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load and prepare data
df = pd.read_csv('geophonesensordata.csv')
features = ['mean', 'top_3_mean', 'min', 'max', 'std_dev', 'median',
            'q1', 'q3', 'skewness', 'dominant_freq', 'energy']
X = df[features]
y = df['activity']

# Encode categorical target
le = LabelEncoder()
y = le.fit_transform(y)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)

# Evaluate
train_score = rf_model.score(X_train_scaled, y_train)
test_score = rf_model.score(X_test_scaled, y_test)
print(f'Train accuracy: {train_score:.3f}')
print(f'Test accuracy: {test_score:.3f}')

# Save model and scaler
joblib.dump(rf_model, 'rf_model.joblib')
joblib.dump(scaler, 'scaler.joblib')
joblib.dump(le, 'label_encoder.joblib')

# Prediction function


def predict_activity(data_dict):
    """
    Predict activity from sensor readings
    data_dict: Dictionary containing sensor readings matching feature names
    """
    # Load saved models
    model = joblib.load('rf_model.joblib')
    scaler = joblib.load('scaler.joblib')
    le = joblib.load('label_encoder.joblib')

    # Prepare input
    input_df = pd.DataFrame([data_dict])
    scaled_input = scaler.transform(input_df)

    # Predict
    prediction = model.predict(scaled_input)
    activity = le.inverse_transform(prediction)[0]

    return activity


# Example usage
sample_data = {
    'mean': 0.5,
    'top_3_mean': 0.8,
    'min': -1,
    'max': 2,
    'std_dev': 0.3,
    'median': 0.4,
    'q1': 0.2,
    'q3': 0.7,
    'skewness': 0.1,
    'dominant_freq': 20.5,
    'energy': 1.2
}

predicted_activity = predict_activity(sample_data)
print(f'Predicted activity: {predicted_activity}')
