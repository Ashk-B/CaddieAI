import cv2
import numpy as np
import pandas as pd


def analyze_swing_video(video_path):
    video = cv2.VideoCapture(video_path)

    if not video.isOpened():
        print("Error: Could not open video.")
        return None

    swing_features = {
        'club_head_speed': None,
        'ball_speed': None,
        'spin_rate': None
    }

    # Frame-based processing logic
    frame_count = 0
    club_movement = []
    ball_movement = []
    previous_frame = None

    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0)

        if previous_frame is None:
            previous_frame = gray_frame
            continue

        frame_delta = cv2.absdiff(previous_frame, gray_frame)
        previous_frame = gray_frame

        _, thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            if cv2.contourArea(contour) < 500:
                continue
            (x, y, w, h) = cv2.boundingRect(contour)

            # Detect club movement
            if w > 20 and h > 50:
                club_movement.append((x, y, w, h))

            # Detect ball movement
            if w < 50 and h < 50 and club_movement:
                ball_movement.append((x, y, w, h))

        frame_count += 1

    video.release()
    cv2.destroyAllWindows()

    if club_movement:
        swing_features['club_head_speed'] = calculate_speed(club_movement)
    if ball_movement:
        swing_features['ball_speed'] = calculate_speed(ball_movement)
    if swing_features['ball_speed'] is not None:
        swing_features['spin_rate'] = estimate_spin_rate(swing_features['ball_speed'])

    return swing_features


def calculate_speed(movement):
    start_pos = movement[0]
    end_pos = movement[-1]
    distance_moved = np.sqrt((end_pos[0] - start_pos[0]) ** 2 + (end_pos[1] - start_pos[1]) ** 2)
    frame_rate = 30  # Assuming 30 FPS
    speed = distance_moved / frame_rate
    return speed * 3  # Approximation for mph or kph


def estimate_spin_rate(ball_speed):
    return 5000 / ball_speed  # Simplified formula


def save_swing_data(swing_features, output_path="swing_data.csv"):
    df = pd.DataFrame([swing_features])
    df.to_csv(output_path, index=False)
    print(f"Swing data saved to {output_path}")
