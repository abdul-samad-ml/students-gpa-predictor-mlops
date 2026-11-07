from fastapi import FastAPI
import joblib
from pydantic import BaseModel
app=FastAPI()
from sklearn.preprocessing import StandardScaler
scaler=joblib.load('gpa_scaler.pkl')
model=joblib.load('ridge_gpa_model.pkl')
class StudentsData(BaseModel):
    gpa_semester_1:float
    attendance_rate:float
    assignments_submitted_rate:float
    lms_logins_per_week:float
    previous_qualification_grade:float
    admission_grade:float

@app.post('/predict')
def predict_gpa(data:StudentsData):
    grade_trend=data.previous_qualification_grade-data.admission_grade
    previous_academic_strenght=(data.previous_qualification_grade+data.admission_grade)/2

    features=[[
    data.gpa_semester_1,
    data.attendance_rate,
    data.assignments_submitted_rate,
    data.lms_logins_per_week,
    grade_trend,
    previous_academic_strenght
     
     ]]
    scaled_features=scaler.transform(features)
    predictions=model.predict(scaled_features)
    return {"Predicted GPA : ":float(predictions[0])}