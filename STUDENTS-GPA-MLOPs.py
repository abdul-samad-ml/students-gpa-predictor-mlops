# This will Predicts Studnts GPA Based Upon Linear Regression Algorithim of ML 
import pandas as pd 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
import joblib
dt=pd.read_csv('STUDENTS-GRADE-ML-PROJECT/student_dropout_risk_2026.csv')
dt=dt.dropna()
dt=pd.get_dummies(dt,dtype=int)
dt['grade_trend'] = dt['previous_qualification_grade'] - dt['admission_grade']
dt['previous_academic_strenght'] =  (dt['previous_qualification_grade'] + dt['admission_grade'])/2

selected_features=[

    'gpa_semester_1', 'attendance_rate','assignments_submitted_rate',

      'lms_logins_per_week', 'grade_trend',

        'previous_academic_strenght'

]
X=dt[selected_features]
y=dt['gpa_semester_2']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
scaler=StandardScaler()
X_train_scaled=scaler.fit_transform(X_train)
X_test_scaled=scaler.transform(X_test)
final_model=Ridge(
    alpha=0.1,
    random_state=42
)
final_model.fit(X_train_scaled,y_train)
predictions=final_model.predict(X_test_scaled)
#mae=mean_absolute_error(y_test,predictions)
#print(f"The Mean Absolute Error is : {mae}")
#print("The Exact 10 GPA Predictions : ",predictions[:10])
joblib.dump(final_model,'ridge_gpa_model.pkl')
joblib.dump(scaler,'gpa_scaler.pkl')