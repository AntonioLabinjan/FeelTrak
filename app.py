from flask import Flask, render_template, request, jsonify
from deepface import DeepFace
import cv2
import numpy as np
import base64
import pandas as pd
import random

app = Flask(__name__)

# Load the CSV file into a DataFrame
df = pd.read_csv("C:/Users/Korisnik/Desktop/CV_MUSIC/face_emotion_analysis/data_moods (1) (1).csv")


def recommend_song(mood):
    print(f"Received mood for recommendation: {mood}")
    filtered_songs = df[df['mood'].str.strip().str.lower() == mood.lower()]
    print(f"Filtered songs based on mood: {filtered_songs}")

    if filtered_songs.empty:
        return 'No songs found for this mood.'
    song = filtered_songs.sample().iloc[0]
    return {
        'name': song['name'],
        'artist': song['artist'],
        'album': song['album'],
        'release_date': song['release_date']
    }

# Route for the homepage
@app.route('/')
def index():
    return render_template('index.html')

# Route to analyze frame sent from the frontend
import traceback  # Add this to capture the full error stack trace

@app.route('/analyze_frame', methods=['POST'])
def analyze_frame():
    try:
        # Get the base64 encoded image from the request
        data_url = request.json['image']
        # Strip the metadata from the base64 URL
        encoded_image = data_url.split(",")[1]
        
        # Decode the base64 image to bytes
        img_data = base64.b64decode(encoded_image)
        np_arr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        # Analyze the frame using DeepFace
        analysis = DeepFace.analyze(img_path=img, actions=['emotion'])
        
        # Since the result might be a list, handle it accordingly
        if isinstance(analysis, list):
            # If multiple faces, select the first one
            analysis = analysis[0]
        
        dominant_emotion = analysis['dominant_emotion']
        
        # Get a recommended song based on the detected emotion
        recommended_song = recommend_song(dominant_emotion)
        
        return jsonify({'emotion': dominant_emotion, 'recommended_song': recommended_song}), 200
    
    except Exception as e:
        # Log the error details to the console
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(port=4000, debug=True)
