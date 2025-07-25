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
        if not os.path.exists(DATA_PATH):
            raise HTTPException(status_code=400, detail=f"CSV file not found at {DATA_PATH}")
        
        df = pd.read_csv(DATA_PATH)
        
        print(f"DataFrame columns: {df.columns.tolist()}")
        print(f"DataFrame rows: {len(df)}")
        
        if df.empty:
            raise HTTPException(status_code=400, detail="No data found in CSV file")

        required_columns = ['temperature', 'ph_20c', 'density_20c']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise HTTPException(status_code=400, detail=f"Missing columns in CSV: {missing_columns}")
        
        features = df[required_columns].fillna(0)

        base_date = datetime(2024, 1, 1)
        dates = df['days_since_start'].apply(lambda x: (base_date + timedelta(days=int(x))).strftime('%Y-%m-%d')).tolist()

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