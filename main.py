from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os
import shutil
from pathlib import Path
import subprocess
from fastapi.responses import JSONResponse
import pandas as pd
import joblib
from datetime import datetime, timedelta
import numpy as np
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://diary-lab-diego-centella-jalafunds-projects.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = os.getenv("MODEL_PATH", "./acidity_model.pkl")
try:
    model = joblib.load(MODEL_PATH)
except FileNotFoundError:
    raise Exception(f"Model file not found at {MODEL_PATH}")

DATA_PATH = os.getenv("DATA_PATH", "./raw_milk_data.csv")

@app.get("/")
async def root():
    return {"message": "FastAPI microservice is running"}

@app.post("/process-csv")
async def process_csv(file: UploadFile = File(...)):
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are allowed.")
    temp_dir = Path("/tmp/temp")
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_file_path = temp_dir / file.filename
    cleaned_file_path = temp_dir / "fileTest_cleaned.csv"

    try:
        with temp_file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        python_script_path = Path(__file__).parent / "clean_automate_csv.py"
        if not python_script_path.exists():
            raise HTTPException(status_code=500, detail="Python script not found.")

        result = subprocess.run(
            ["python3", str(python_script_path), str(temp_file_path)],
            cwd=temp_dir,
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            raise HTTPException(status_code=500, detail=f"Python script error: {result.stderr}")

        if not cleaned_file_path.exists():
            raise HTTPException(status_code=500, detail="Cleaned CSV file not found.")

        with cleaned_file_path.open("r", encoding="utf-8") as f:
            cleaned_csv_data = f.read()

        return {"cleaned_csv": cleaned_csv_data}
    finally:
        for path in [temp_file_path, cleaned_file_path]:
            if path.exists():
                path.unlink()

@app.get("/predict-acidity")
async def predict_acidity():
    try:
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
                'date': future_date.strftime('%Y-%m-%d'),
                'titratable_acidity_predicted': round(float(acidity_predicted), 3)
            })

        return {"predictions": predictions}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error making predictions: {str(e)}")