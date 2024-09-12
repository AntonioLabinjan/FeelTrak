from flask import Flask, render_template, request, jsonify, redirect
from deepface import DeepFace
import cv2
import numpy as np
import base64
import pandas as pd
import random
from flask_sqlalchemy import SQLAlchemy
import requests
import traceback
import datetime
from datetime import datetime
from sqlalchemy.sql import func
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user


app = Flask(__name__)

# Database Configuration
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///songs.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SECRET_KEY'] = 'secret_369'
db = SQLAlchemy(app)

login_manager = LoginManager()
login_manager.init_app(app)

# Define the User model
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), unique=True, nullable=False)
    password_hash = db.Column(db.String(128), nullable=False)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

# Define the Song model
class Song(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    artist = db.Column(db.String(100), nullable=False)
    album = db.Column(db.String(100), nullable=False)
    release_date = db.Column(db.String(100), nullable=False)
    mood = db.Column(db.String(50), nullable=False)

class RecommendationHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    song_id = db.Column(db.Integer, db.ForeignKey('song.id'), nullable=False)
    recommended_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)

    song = db.relationship('Song', backref=db.backref('recommendations', lazy=True))

# Initialize Flask-Login UserMixin
class UserSession(UserMixin):
    def __init__(self, user_id):
        self.id = user_id

# User loader function
@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Function to load CSV data into the database
def load_csv_data():
    df = pd.read_csv("C:/Users/Korisnik/Desktop/CV_MUSIC/face_emotion_analysis/data_moods (1) (1).csv")
    for _, row in df.iterrows():
        song = Song(name=row['name'], artist=row['artist'], album=row['album'],
                    release_date=row['release_date'], mood=row['mood'].strip().lower())
        db.session.add(song)
    db.session.commit()

# YouTube Data API Key (replace with your actual key)
# PROBLEM: ISTEKA NAN JE API xD
YOUTUBE_API_KEY = 'AIzaSyB-XtO_O3GRPSvjeZgRtqO9nwgJaMxG6fs'
YOUTUBE_SEARCH_URL = 'https://www.googleapis.com/youtube/v3/search'

# Function to fetch YouTube link for the recommended song
def get_youtube_link(song_name, artist):
    search_query = f"{song_name} {artist}"
    params = {
        'part': 'snippet',
        'q': search_query,
        'key': YOUTUBE_API_KEY,
        'type': 'video',
        'maxResults': 1
    }
    
    response = requests.get(YOUTUBE_SEARCH_URL, params=params)
    #print(f"Requesting YouTube search for: {search_query}")
    #print(f"API response: {response.status_code}")
    
    if response.status_code == 200:
        results = response.json().get('items')
        #print(f"Results from YouTube API: {results}")
        if results:
            video_id = results[0]['id']['videoId']
            youtube_link = f"https://www.youtube.com/watch?v={video_id}"  # Updated link format
            #print(f"Found YouTube link: {youtube_link}")
            return youtube_link
        else:
            print("No results found on YouTube.")
    else:
        print(f"Error response from YouTube API: {response.text}")
    
    return None

# Route for the homepage
@app.route('/')
def index():
    return render_template('index.html')

from datetime import datetime

from datetime import datetime, timedelta

def recommend_song(mood):
    print(f"Received mood for recommendation: {mood}")
    mood = mood.lower()
    filtered_songs = Song.query.filter(Song.mood == mood).all()
    print(f"Filtered songs based on mood: {filtered_songs}")

    if not filtered_songs:
        return 'No songs found for this mood.'
    
    song = random.choice(filtered_songs)
    
    # Get the YouTube link
    youtube_link = get_youtube_link(song.name, song.artist)
    
    # Save the recommendation to history with local time adjusted by adding 2 hours
    recommendation = RecommendationHistory(
        song_id=song.id,
        recommended_at=datetime.now()  # Adjusting local time by adding 2 hours
    )
    db.session.add(recommendation)
    db.session.commit()
    
    return {
        'name': song.name,
        'artist': song.artist,
        'album': song.album,
        'release_date': song.release_date,
        'youtube_link': youtube_link
    }



# Route to analyze frame sent from the frontend
@app.route('/analyze_frame', methods=['POST'])
def analyze_frame():
    try:
        # Get the base64 encoded image from the request
        data_url = request.json['image']
        encoded_image = data_url.split(",")[1]
        
        # Decode the base64 image to bytes
        img_data = base64.b64decode(encoded_image)
        np_arr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        # Analyze the frame using DeepFace
        analysis = DeepFace.analyze(img_path=img, actions=['emotion'])
        
        # Handle multiple face results
        if isinstance(analysis, list):
            analysis = analysis[0]
        
        dominant_emotion = analysis['dominant_emotion']
        
        # Get a recommended song based on the detected emotion
        recommended_song = recommend_song(dominant_emotion)
        
        return jsonify({'emotion': dominant_emotion, 'recommended_song': recommended_song}), 200
    
    except Exception as e:
        # Log the error details to the console
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/recommendation_history')
def recommendation_history():
    try:
        recommendations = RecommendationHistory.query.order_by(RecommendationHistory.recommended_at.desc()).all()
        recommended_songs = []

        for rec in recommendations:
            song = Song.query.get(rec.song_id)
            recommended_songs.append({
                'name': song.name,
                'artist': song.artist,
                'album': song.album,
                'release_date': song.release_date,
                'recommended_at': rec.recommended_at.strftime('%Y-%m-%d %H:%M:%S')
            })

        return render_template('recommendation_history.html', recommended_songs=recommended_songs)

    except Exception as e:
        # Log the error details to the console
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

# Route for mood playlists
@app.route('/mood_playlists')
def mood_playlists():
    # Get all distinct moods from the Song table
    distinct_moods = Song.query.with_entities(Song.mood).distinct().all()
    
    mood_playlists = {}
    
    # For each mood, fetch 10 random songs
    for mood_tuple in distinct_moods:
        mood = mood_tuple[0]  # Extract the mood from the tuple
        songs = Song.query.filter_by(mood=mood).order_by(func.random()).limit(10).all()
        
        # Prepare song data without YouTube links
        song_data = []
        for song in songs:
            song_data.append({
                'name': song.name,
                'artist': song.artist,
                'album': song.album,
                'release_date': song.release_date
            })
        
        # Add songs to the mood playlist
        mood_playlists[mood] = song_data

    return render_template('mood_playlists.html', mood_playlists=mood_playlists)



# Admin login route
@app.route('/admin_login', methods=['GET', 'POST'])
def admin_login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        user = User.query.filter_by(username=username).first()
        if user and user.check_password(password):
            login_user(UserSession(user.id))
            return redirect('/admin')

        return 'Invalid username or password.'

    return render_template('admin_login.html')

# Admin interface route to add new songs
@app.route('/admin', methods=['GET', 'POST'])
@login_required
def admin():
    if request.method == 'POST':
        name = request.form.get('name')
        artist = request.form.get('artist')
        album = request.form.get('album')
        release_date = request.form.get('release_date')
        mood = request.form.get('mood')

        # Create a new song record
        song = Song(name=name, artist=artist, album=album,
                    release_date=release_date, mood=mood.lower().strip())
        db.session.add(song)
        db.session.commit()

        return 'Song added successfully!'

    return render_template('admin.html')

# Admin logout route
@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect('/admin_login')

# CLI command to create an admin user
@app.cli.command('create_admin')
def create_admin():
    username = input('Enter admin username: ')
    password = input('Enter admin password: ')
    hashed_password = generate_password_hash(password)
    admin = User(username=username, password_hash=hashed_password)
    db.session.add(admin)
    db.session.commit()
    print('Admin user created.')



if __name__ == '__main__':
    with app.app_context():
        db.create_all()  # This will create the tables in the database
        if Song.query.count() == 0:  # Only load data if the table is empty
            load_csv_data()
    app.run(port=4000, debug=True)
