# Objective  
Develop a machine learning system to predict whether an employee will enroll in an insurance plan based on demographic and job-related features. This involves:  

- Data preprocessing  
- Model training  
- Building an API for predictions  

## Dataset Overview  
The dataset contains fields such as:  

- `age`: int  
- `gender`: categorical  
- `marital_status`: categorical  
- `salary`: float  
- `employment_type`: categorical  
- `region`: categorical  
- `has_dependents`: categorical (Yes/No)  
- `tenure_years`: int  
- `enrolled`: target (1/0)  

## Data Processing  
Steps include:  

- Handling categorical features using OneHotEncoding  
- Normalizing numerical features with StandardScaler  
- Pipeline built using ColumnTransformer  

## Model  
We selected `RandomForestClassifier` for its ability to handle heterogeneous features and robustness to outliers.  

### Why Random Forest?  
- Handles mixed data types  
- Low tuning needed  
- Feature importance interpretation  

## Evaluation  
The model was evaluated using:  

- Accuracy  
- Precision, Recall, F1-Score  
- ROC-AUC  

It achieved good performance (~85% accuracy) on the test set.  

## API Design  
Built using FastAPI, which provides:  

- Fast performance  
- Built-in Swagger UI  
- Easy schema validation with Pydantic  

### Endpoint: `/predict`  
- Accepts JSON with employee details  
- Returns prediction and enrollment probability  

## Results  
The deployed model provides real-time predictions and is extensible for batch input, logging, and monitoring.  

## Future Improvements  
- Add database logging (e.g., SQLite/PostgreSQL)  
- Enable batch prediction via CSV upload  
- Improve model via hyperparameter tuning or boosting  
- Add CI/CD and Docker deployment  

## Conclusion  
This project successfully demonstrates building a complete ML pipeline — from data to deployment — and shows how AI can assist in real-world decision making for HR/insurance tech.  