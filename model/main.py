from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from caddie_inference import load_caddie_model, predict_club

app = FastAPI(title="CaddieAI - Golf Recommendation Service")

# load the model & encoders once, at startup
model, bundle = load_caddie_model(model_path="caddie_model.pt", encoders_path="encoders.pkl")

# define a model for the input data to validate incoming request
class GolfShotRequest(BaseModel):
    # replace or add fields as they exist in the training set
    distance_to_pin: float
    wind_speed: float
    grass_condition: str
    skill_level: str
    # add more fields later prolly


@app.post("/recommend_club")
def recommend_club_endpoint(request: GolfShotRequest):
    # convert request object to dict
    user_data = request.dict()

    # check that all features are present in user_data
    for col in bundle["feature_cols"]:
        if col not in user_data:
            # If needed, you can set default values or raise an error
            raise HTTPException(
                status_code=400,
                detail=f"Missing required feature: {col}"
            )

    # predict the club using inference method
    recommended_club = predict_club(user_data, model, bundle)

    return {"recommended_club": recommended_club}

if __name__ == "__main__":
    # for local dev: run "python main.py"
    uvicorn.run(app, host="0.0.0.0", port=8000)
