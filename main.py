from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os
import shutil
from pathlib import Path
import subprocess
from fastapi.responses import JSONResponse
import pandas as pd
import joblib
from datetime import datetime
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
model = joblib.load(MODEL_PATH)

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
        df = pd.read_csv(DATA_PATH)
        
        features = df[['evening_temperature', 'ph_20c_evening', 'fat_content_evening']].fillna(0)
        dates = df['date'].tolist()
        

        predictions = model.predict(features)

        formatted_predictions = [
            {
                "date": date,
                "titratable_acidity_predicted": float(pred)
            }
            for date, pred in zip(dates, predictions)
        ]
        
        return {"predictions": formatted_predictions}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error making predictions: {str(e)}")