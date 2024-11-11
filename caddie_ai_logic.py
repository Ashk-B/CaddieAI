import pickle
from sqlalchemy.orm import sessionmaker
from database_setup import Player, InstructionalData, engine
from external_apis import get_weather_data, get_user_location
import traceback

# Load the AI model from the trained model file
def load_ai_model(model_path="models/golf_shot_model.pkl"):
    try:
        with open(model_path, "rb") as model_file:
            model = pickle.load(model_file)
        print("AI model loaded successfully.")
        return model
    except Exception as e:
        print(f"Error loading the AI model: {e}")
        traceback.print_exc()
        return None

# Create a session for database queries
Session = sessionmaker(bind=engine)
session = Session()

# Fetch player data from the database
def get_player_data(player_name):
    try:
        player = session.query(Player).filter_by(name=player_name).first()
        if player:
            print(f"Player data for {player_name} loaded successfully.")
            return player
        else:
            print(f"No data found for player {player_name}.")
            return None
    except Exception as e:
        print(f"Error retrieving player data for {player_name}: {e}")
        traceback.print_exc()
        return None

# Fetch instructional advice from the database
def get_instructional_advice():
    try:
        advice_entries = session.query(InstructionalData).all()
        if advice_entries:
            print("Instructional advice loaded successfully.")
            return advice_entries
        else:
            print("No instructional advice found.")
            return None
    except Exception as e:
        print(f"Error retrieving instructional advice: {e}")
        traceback.print_exc()
        return None

# Function to fetch real-time weather conditions
def get_real_time_conditions():
    try:
        # Fetch GPS location
        location = get_user_location()

        # Fetch weather conditions based on location
        weather = get_weather_data(location['latitude'], location['longitude'])

        print("Real-time conditions fetched successfully.")
        return location, weather
    except Exception as e:
        print(f"Error fetching real-time conditions: {e}")
        traceback.print_exc()
        return None, None

# Make a recommendation based on inputs
def recommend_club(model, player_data, yardage, weather_conditions):
    try:
        # Prepare the input data for the model (you can adjust the features as needed)
        input_data = [
            yardage,
            weather_conditions['wind_speed'],
            weather_conditions['humidity'],
            player_data.club_head_speed,
            player_data.ball_speed,
            player_data.spin_rate
        ]

        # Model expects a 2D array (reshape the input to fit this format)
        input_data = [input_data]

        # Use the model to predict the recommended club
        predicted_club = model.predict(input_data)

        print(f"Recommended club for the shot: {predicted_club[0]}")
        return predicted_club[0]
    except Exception as e:
        print(f"Error making club recommendation: {e}")
        traceback.print_exc()
        return None

# Main function to run the AI caddie logic
def run_caddie_ai(player_name, yardage):
    try:
        # Load the AI model
        model = load_ai_model()

        if not model:
            print("AI model could not be loaded. Exiting.")
            return

        # Get player data
        player_data = get_player_data(player_name)
        if not player_data:
            print(f"No data available for player {player_name}. Exiting.")
            return

        # Get real-time conditions (weather, GPS)
        location, weather_conditions = get_real_time_conditions()
        if not weather_conditions:
            print("Real-time conditions could not be retrieved. Exiting.")
            return

        # Recommend the best club based on the inputs
        recommended_club = recommend_club(model, player_data, yardage, weather_conditions)
        if recommended_club:
            print(f"Recommended Club: {recommended_club}")

        # retrieve instructional advice
        advice = get_instructional_advice()
        if advice:
            print("\nInstructional Advice from Experts:")
            for entry in advice:
                print(f"- From {entry.source}: {entry.advice}")

    except Exception as e:
        print(f"An error occurred while running the AI caddie: {e}")
        traceback.print_exc()

# Example usage
if __name__ == "__main__":
    player_name = "John Doe"  # Replace with actual player name
    yardage = 150  # Replace with the yardage from the hole
    run_caddie_ai(player_name, yardage)
