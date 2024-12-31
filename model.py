import pandas as pd
import joblib


class SensorPredictor:
    def __init__(self, model_path='rf_model.joblib',
                 scaler_path='scaler.joblib',
                 encoder_path='label_encoder.joblib'):
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        self.le = joblib.load(encoder_path)

    def predict(self, sensor_data):
        df = pd.DataFrame([sensor_data])
        scaled = self.scaler.transform(df)
        pred = self.model.predict(scaled)
        return self.le.inverse_transform(pred)[0]
