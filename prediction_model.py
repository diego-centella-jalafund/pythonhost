import pandas as pd
import joblib
from datetime import datetime, timedelta
import numpy as np

model = joblib.load('acidity_model.pkl')

last_date = datetime.now().date()
reference_date = pd.to_datetime('2024-04-14').date()
days_since_start = (last_date - reference_date).days

avg_temperature = 20.0 
avg_ph_20c = 6.7        
avg_density_20c = 1.030  

predictions = []
for i in range(1, 21):
    future_day = days_since_start + i
    future_date = last_date + timedelta(days=i)
    temperature = np.clip(avg_temperature + np.random.uniform(-0.5, 0.5), 15, 25)
    ph_20c = np.clip(avg_ph_20c + np.random.uniform(-0.02, 0.02), 6.60, 6.80)
    density_20c = np.clip(avg_density_20c + np.random.uniform(-0.001, 0.001), 1.028, 1.034)


    future_data = pd.DataFrame({
        'days_since_start': [future_day],
        'temperature': [temperature],
        'ph_20c': [ph_20c],
        'density_20c': [density_20c]
    })

    acidity_predicted = model.predict(future_data)[0]

    acidity_predicted = np.clip(acidity_predicted, 0.13, 0.18)

    predictions.append({
        'date': future_date,
        'titratable_acidity_predicted': round(acidity_predicted, 3)
    })

for pred in predictions:
    print(f"Date: {pred['date']}, Titratable Acidity Predicted: {pred['titratable_acidity_predicted']}")