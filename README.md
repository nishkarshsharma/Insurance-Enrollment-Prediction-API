# NeuStackApp

## Project Overview
This project is an end-to-end machine learning pipeline and web API to predict whether an employee will enroll in an insurance plan based on their demographic and employment information.

## Problem Statement
Given employee information (age, gender, salary, etc.), we aim to predict whether the employee will enroll in the company insurance scheme. This is a binary classification problem.

## Project Structure
```
NeuStackApp/
├── main.py                  # FastAPI app with predict endpoint
├── train_pipeline.py        # Model training pipeline
├── model.joblib             # Trained ML model (after running training script)
├── requirements.txt         # Python dependencies
├── README.md                # Project setup and usage instructions
├── report.md                # Detailed project report
```

## Setup Instructions

### Clone the repository
```bash
git clone https://github.com/nishkarshsharma/neustackapp.git
cd neustackapp
```

### Create virtual environment
```bash
python -m venv venv
source venv/bin/activate
```

### Install dependencies
```bash
pip install -r requirements.txt
```

### Train the model
```bash
python train_pipeline.py
```
This creates `model.joblib` — a serialized pipeline with preprocessing and model.

### Run the FastAPI app
```bash
uvicorn main:app --reload
```
Visit the docs at: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

### Example API Request
**POST /predict**
```json
{
    "age": 35,
    "gender": "Male",
    "marital_status": "Single",
    "salary": 72000,
    "employment_type": "Full-Time",
    "region": "West",
    "has_dependents": "Yes",
    "tenure_years": 4
}
```

### Sample Response
```json
{
    "enrolled": true,
    "probability": 0.82
}
```

## Tech Stack
- Python
- FastAPI
- scikit-learn
- Pandas
- joblib  