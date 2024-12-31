# GeoSensor-Movement-Predictor
Linear Regression ML Model To Predict Type Of Movement Based On GeoSensor Dataset (walking, running.. )

- Dataset included as CSV
- Model trainer script included
- Model class included
- Predictor script included
- Samples usage included in predictor script

## How To Use

### Setup

- Intall pyenv: https://github.com/pyenv/pyenv?tab=readme-ov-file#installation

Run the following commands in terminal:

```bash

cd ~ && git clone https://github.com/CYFARE/GeoSensor-Movement-Predictor && cd GeoSensor-Movement-Predictor
pyenv install 3.11.11
pyenv local 3.11.11
python -V && python -m venv venv
source venv/bin/activate
python -m pip install -r requirements.txt
```
- Run: `python predictor.py` to see model prediction results with sample data provided in the same python script. 

### Prediction of Live Data

Given types of geosensor data, predict.py can be used to determine at real-time speeds the type of movement - walking, running etc. based on trained dataset. It's trained using linear regression so it's quite accurate.
