from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

# import inference utilities
from caddie_inference import load_caddie_model, predict_club

app = FastAPI(title="CaddieAI - Golf Recommendation Service")

# 1) Load the model & encoders once, at startup
model, bundle = load_caddie_model(model_path="caddie_model.pt", encoders_path="encoders.pkl")

# 2) Define a Pydantic model for the input data
#    So you can validate incoming request fields
class GolfShotRequest(BaseModel):
    # Replace or add fields as they exist in your training set
    distance_to_pin: float
    wind_speed: float
    grass_condition: str
    skill_level: str
    # Add more fields if your model uses them
    # e.g. wind_direction, temperature, etc.

@app.post("/recommend_club")
def recommend_club_endpoint(request: GolfShotRequest):
    # Convert request object to dict
    user_data = request.dict()

    # Check that all features are present in user_data
    # Optionally handle missing fields, etc.
    for col in bundle["feature_cols"]:
        if col not in user_data:
            # If needed, you can set default values or raise an error
            raise HTTPException(
                status_code=400,
                detail=f"Missing required feature: {col}"
            )

    # 3) Predict the club using our inference method
    recommended_club = predict_club(user_data, model, bundle)

    return {"recommended_club": recommended_club}

if __name__ == "__main__":
    # For local dev: run "python main.py"
    uvicorn.run(app, host="0.0.0.0", port=8000)
