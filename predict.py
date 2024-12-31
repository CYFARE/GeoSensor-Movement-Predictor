# predict.py
from model import SensorPredictor

predictor = SensorPredictor()

# Example live data processing
def process_live_data(sensor_readings):
    data = {
        'mean': sensor_readings['mean'],
        'top_3_mean': sensor_readings['top_3_mean'],
        'min': sensor_readings['min'],
        'max': sensor_readings['max'],
        'std_dev': sensor_readings['std_dev'],
        'median': sensor_readings['median'],
        'q1': sensor_readings['q1'],
        'q3': sensor_readings['q3'],
        'skewness': sensor_readings['skewness'],
        'dominant_freq': sensor_readings['dominant_freq'],
        'energy': sensor_readings['energy']
    }
    return predictor.predict(data)

# Example usage
sample_data = {
    'mean': 0.5, 'top_3_mean': 0.8, 'min': -1,
    'max': 2, 'std_dev': 0.3, 'median': 0.4,
    'q1': 0.2, 'q3': 0.7, 'skewness': 0.1,
    'dominant_freq': 20.5, 'energy': 1.2
}

print(f"Predicted activity: {process_live_data(sample_data)}")
