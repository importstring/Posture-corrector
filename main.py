import tkinter as tk
from tkinter import messagebox
import cv2
import mediapipe as mp
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle
import time

# Define constants for actions
ACTIONS = ["lean_forward", "lean_backward", "tilt_head_up", "tilt_head_down", "tilt_head_side", "tilt_head_other_side", "knees_up", "good_posture", "wrists_bent"]

# Define function to collect data
def collect_data(action):
    messagebox.showinfo("Data Collection", f"Please record {action} posture.")
    cap = cv2.VideoCapture(0)
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)
    data = []
    start_time = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            features = [landmark.x for landmark in landmarks] + [landmark.y for landmark in landmarks] + [landmark.z for landmark in landmarks]
            data.append(features)
        cv2.imshow('Collecting data for ' + action, frame)
        if time.time() - start_time > 5:  # collect data for 5 seconds
            break
        if cv2.waitKey(5) & 0xFF == 27:
            break
    cap.release()
    cv2.destroyAllWindows()
    elapsed_time = time.time() - start_time
    messagebox.showinfo("Data Collection", f"Data collection for {action} completed in {elapsed_time:.2f} seconds.")
    return data

# Define function to train model
def train_model(data, labels):
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
    model = LogisticRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Model accuracy:", accuracy)
    return model

# Define function to send notification
def send_notification(message):
    messagebox.showinfo("Notification", message)

# Define main function
def main():
    # Initialize Tkinter
    tk.Tk().withdraw()  # Prevents the Tkinter window from appearing

    # Collect data for each action
    data = {}
    for action in ACTIONS:
        print("Collecting data for", action)
        data[action] = collect_data(action)

    # Train model
    labels = []
    for action in ACTIONS:
        labels.extend([action] * len(data[action]))
    data = [item for sublist in data.values() for item in sublist]
    model = train_model(data, labels)

    # Save model
    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)

    # Start monitoring
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)
    last_notification_time = time.time()  # Initialize last notification time
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            features = [landmark.x for landmark in landmarks] + [landmark.y for landmark in landmarks] + [landmark.z for landmark in landmarks]
            prediction = model.predict_proba([features])
            if prediction[0].max() > 0.8 and prediction[0].argmax() != ACTIONS.index("good_posture"):  # Check if prediction is almost certain
                current_time = time.time()
                if current_time - last_notification_time > 30:  # Check if 30 seconds have passed since last notification
                    send_notification("Please adjust your posture")
                    last_notification_time = current_time  # Update last notification time
        cv2.imshow('Monitoring', frame)
        if cv2.waitKey(5) & 0xFF == 27:
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
