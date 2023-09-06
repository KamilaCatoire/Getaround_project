import gc
import mlflow
import uvicorn
import numpy as np
import pandas as pd
from pydantic import BaseModel
from typing import Literal, List, Union
from fastapi import FastAPI, File, UploadFile
from fastapi.encoders import jsonable_encoder
from fastapi.responses import RedirectResponse

description = """
Welcome to my rental price predictor API !\n
Submit the characteristics of your car and a Machine Learning model, trained on GetAround data, will recommend you a price per day for your rental. 

**Use the endpoint `/predict` to estimate the daily rental price of your car !**
"""

tags_metadata = [
    {
        "name": "Predictions",
        "description": "Use this endpoint for getting predictions"
    }
]

app = FastAPI(
    title="💸 Car Rental Price Predictor",
    description=description,
    version="0.1",
    openapi_tags=tags_metadata
)

class Car(BaseModel):
    model_key: Literal['Citroën','Peugeot','PGO','Renault','Audi','BMW','Mercedes','Opel','Volkswagen','Ferrari','Mitsubishi','Nissan','SEAT','Subaru','Toyota','other'] 
    mileage: Union[int, float]
    engine_power: Union[int, float]
    fuel: Literal['diesel','petrol','other']
    paint_color: Literal['black','grey','white','red','silver','blue','beige','brown','other']
    car_type: Literal['convertible','coupe','estate','hatchback','sedan','subcompact','suv','van']
    private_parking_available: bool
    has_gps: bool
    has_air_conditioning: bool
    automatic_car: bool
    has_getaround_connect: bool
    has_speed_regulator: bool
    winter_tires: bool

# Redirect automatically to /docs (without showing this endpoint in /docs)
@app.get("/", include_in_schema=False)
async def docs_redirect():
    return RedirectResponse(url='/docs')

# Make predictions
@app.post("/predict", tags=["Predictions"])
async def predict(cars: List[Car]):
    # clean unused memory
    gc.collect(generation=2)

    # Read input data
    car_features = pd.DataFrame(jsonable_encoder(cars))

    # Load model as a PyFuncModel.
    logged_model = 'runs:/78ffa628d149410a815b2e5ab880e6ad/model'
    loaded_model = mlflow.pyfunc.load_model(logged_model)

    # Predict and format response
    prediction = loaded_model.predict(car_features)
    response = {"prediction": prediction.tolist()}
    return response

if __name__=="__main__":
    uvicorn.run(app, host="0.0.0.0", port=4000, debug=True, reload=True)
