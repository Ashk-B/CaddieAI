import pickle
from sqlalchemy.orm import sessionmaker
from database_setup import Player, InstructionalData, engine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from video_processing_instructional import process_instructional_video
import traceback

# Create a session for database queries
Session = sessionmaker(bind=engine)
session = Session()

# Load player data from the database
def get_player_data():
    try:
        players = session.query(Player).all()
        data = []
        for player in players:
            # Extract player features for the model
            player_data = {
                'handicap': player.handicap,
                'average_yardages': player.average_yardages,
                'club_head_speed': player.club_head_speed,
                'ball_speed': player.ball_speed,
                'spin_rate': player.spin_rate
            }
            data.append(player_data)
        return data
    except Exception as e:
        print(f"Error loading player data: {e}")
        traceback.print_exc()
        return None


# Load instructional advice from the database
def get_instructional_advice():
    try:
        advice_entries = session.query(InstructionalData).all()
        instructional_data = {}
        for entry in advice_entries:
            instructional_data[entry.source] = entry.advice
        return instructional_data
    except Exception as e:
        print(f"Error loading instructional advice: {e}")
        traceback.print_exc()
        return None


# Prepare and combine all data sources
def prepare_data(player_data, instructional_data):
    try:
        # Create a list of numeric data from player data
        X_data = []
        y_labels = []

        for player in player_data:
            # Example: Combine average yardages and swing features for each player
            for club, yardage in player['average_yardages'].items():
                X_data.append([
                    yardage,
                    player['club_head_speed'],
                    player['ball_speed'],
                    player['spin_rate']
                ])
                y_labels.append(club)  # Club used for the yardage

        # Instructional data can be incorporated as additional features or metadata

        return X_data, y_labels
    except Exception as e:
        print(f"Error preparing data: {e}")
        traceback.print_exc()
        return None, None


# Train a model dynamically based on the prepared data
def train_golf_model(X, y):
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        accuracy = model.score(X_test, y_test)
        print(f"Model Accuracy: {accuracy * 100:.2f}%")
        return model
    except Exception as e:
        print(f"Error training the model: {e}")
        traceback.print_exc()
        return None


# Save the trained model
def save_model(model, file_path="models/golf_shot_model.pkl"):
    try:
        with open(file_path, "wb") as model_file:
            pickle.dump(model, model_file)
        print(f"Model saved to {file_path}")
    except Exception as e:
        print(f"Error saving the model: {e}")
        traceback.print_exc()


# Full training pipeline
if __name__ == "__main__":
    try:
        # Load player data from the database
        print("Loading player data from the database...")
        player_data = get_player_data()

        if player_data is None:
            print("No player data found. Exiting.")
            exit()

        # Load instructional advice from the database
        print("Loading instructional advice from the database...")
        instructional_data = get_instructional_advice()

        # Process an instructional video and store advice (optional, if new videos need to be processed)
        video_path = "videos/instructional_video.mp4"
        process_instructional_video(video_path)

        # Prepare and combine all the data
        X, y = prepare_data(player_data, instructional_data)

        if X is not None and y is not None:
            # Train the model
            golf_model = train_golf_model(X, y)

            if golf_model:
                # Save the trained model
                save_model(golf_model)
    except Exception as e:
        print(f"An error occurred: {e}")
        traceback.print_exc()
