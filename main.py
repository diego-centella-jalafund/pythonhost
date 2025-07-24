from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os
import shutil
from pathlib import Path
import subprocess

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://diary-lab-diego-centella-jalafunds-projects.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

        python_script_path = Path("clean_automate_csv.py")
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