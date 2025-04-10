import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

# Define the input data schema
class EmployeeData(BaseModel):
    age: int
    gender: str
    marital_status: str
    salary: float
    employment_type: str
    region: str
    has_dependents: str
    tenure_years: int

@app.post("/predict")
def predict_enrollment(data: EmployeeData):
    try:
        # Convert input to DataFrame (this is key!)
        input_data = pd.DataFrame([data.dict()])

        # Load model
        model = joblib.load("model.joblib")

        # Predict
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]

        return {
            "enrolled": bool(prediction),
            "probability": round(probability, 2)
        }

    except Exception as e:
        print("ERROR:", e)
        raise HTTPException(status_code=500, detail=str(e))
