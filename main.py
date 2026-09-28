import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Literal, List
from fastapi.middleware.cors import CORSMiddleware

from scoring_rules import normalize_country, range_notes, day_error

model = joblib.load('Mental_Health_Model.pkl')

app = FastAPI(title="Manas Meter API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


#A first Pydantic Model
class StudentData(BaseModel):
    age                     : int = Field(..., ge=10, le=100)
    gender                  : Literal['Male', 'Female']
    country                 : str = Field(..., min_length=1)
    academic_level          : Literal['Undergraduate', 'Graduate', 'High School']
    most_used_platform      : Literal['Facebook', 'LinkedIn', 'Instagram', 'Snapchat','Twitter','YouTube', 'TikTok', 'LINE', 'KakaoTalk', 'VKontakte', 'WhatsApp','WeChat']
    purpose_of_use          : Literal['Networking', 'Education', 'Entertainment', 'News']
    avg_daily_usage_hours   : float = Field(..., ge=0, le=24)
    daily_unlocks           : int   = Field(..., ge=0)
    study_hours             : float = Field(..., ge=0, le=24)
    physical_activity_hours : float = Field(..., ge=0, le=24)
    sleep_hours_per_night   : float = Field(..., ge=0, le=24)
    stress_level            : Literal['Medium', 'Low', 'Very High', 'High']


# Describe what we send back
class PredictionResponse(BaseModel):
    predicted_mental_health_score: float
    notes: List[str] = []   # plain-language cautions, e.g. inputs outside the trained range


@app.get('/')
def greet():
    return {"message": "Manas Meter API is running"}


@app.post('/predict', response_model=PredictionResponse)
def predict(data: StudentData):

   problem = day_error(data.sleep_hours_per_night, data.study_hours, data.physical_activity_hours)
   if problem:
       raise HTTPException(status_code=422, detail=problem)

   country_group = normalize_country(data.country)

   input_row = pd.DataFrame([{
        'Age'                       :data.age,
        'Gender'                    :data.gender,
        'Country'                   :data.country,
        'Academic_Level'            :data.academic_level,
        'Most_Used_Platform'        :data.most_used_platform,
        'Purpose_Of_Use'            :data.purpose_of_use,
        'Avg_Daily_Usage_Hours'     :data.avg_daily_usage_hours,
        'Daily_Unlocks'             :data.daily_unlocks,
        'Study_Hours'               :data.study_hours,
        'Physical_Activity_Hours'   :data.physical_activity_hours,
        'Sleep_Hours_Per_Night'     :data.sleep_hours_per_night,
        'Stress_Level'              :data.stress_level,
        'Grouped_country'           :country_group
   }])

   prediction = float(model.predict(input_row)[0])
   prediction = min(10.0, max(0.0, prediction))

   return PredictionResponse(
       predicted_mental_health_score=round(prediction, 2),
       notes=range_notes(data.model_dump()),
   )
