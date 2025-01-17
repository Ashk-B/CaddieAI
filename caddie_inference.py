import torch
import torch.nn as nn
import numpy as np
import joblib

# reuse the same network architecture from train_model.py for consistent model loading
from train_model import CaddieNet

def load_caddie_model(
    model_path="caddie_model.pt",
    encoders_path="encoders.pkl",
    hidden_dim=32
):
    # Load encoders
    bundle = joblib.load(encoders_path)
    club_label_encoder = bundle["club_label_encoder"]
    feature_encoders = bundle["feature_encoders"]
    scaler = bundle["scaler"]
    feature_cols = bundle["feature_cols"]

    # Rebuild the model
    input_dim = len(feature_cols)
    output_dim = len(club_label_encoder.classes_)
    model = CaddieNet(input_dim, hidden_dim, output_dim)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    return model, bundle

def predict_club(user_data: dict, model, bundle):
    """
    user_data: a dict that contains keys matching `bundle["feature_cols"]`.
      Example:
      {
        "distance_to_pin": 150,
        "wind_speed": 10,
        "grass_condition": "wet",
        "skill_level": "Advanced",
        ...
      }

    Returns: predicted club name (string).
    """
    club_label_encoder = bundle["club_label_encoder"]
    feature_encoders = bundle["feature_encoders"]
    scaler = bundle["scaler"]
    feature_cols = bundle["feature_cols"]

    # Build input vector
    X_list = []
    for col in feature_cols:
        val = user_data[col]
        if col in feature_encoders:
            le = feature_encoders[col]
            val = le.transform([str(val)])[0]
        else:
            val = float(val)
        X_list.append(val)

    # Convert to numpy
    X_array = np.array(X_list, dtype=np.float32).reshape(1, -1)

    # Scale the input
    X_scaled = scaler.transform(X_array)
    X_t = torch.from_numpy(X_scaled)

    # Forward pass
    with torch.no_grad():
        outputs = model(X_t)
        _, pred_idx = torch.max(outputs, 1)

    recommended_club = club_label_encoder.inverse_transform([pred_idx.item()])[0]
    return recommended_club
