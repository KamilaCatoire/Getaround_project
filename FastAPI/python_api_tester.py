import requests

response = requests.post("https://fastapi-ga-kc-dd15d3a6caeb.herokuapp.com/predict", json=
[
  {
    "model_key": "Citroën",
    "mileage": 100000,
    "engine_power": 100,
    "fuel": "diesel",
    "paint_color": "black",
    "car_type": "convertible",
    "private_parking_available": True,
    "has_gps": True,
    "has_air_conditioning": True,
    "automatic_car": True,
    "has_getaround_connect": True,
    "has_speed_regulator": True,
    "winter_tires": True
  },
  {
    "model_key": "Audi",
    "mileage": 100000,
    "engine_power": 150,
    "fuel": "diesel",
    "paint_color": "black",
    "car_type": "convertible",
    "private_parking_available": True,
    "has_gps": True,
    "has_air_conditioning": True,
    "automatic_car": True,
    "has_getaround_connect": True,
    "has_speed_regulator": True,
    "winter_tires": True
  }
]
)
print(response.json())