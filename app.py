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
from datetime import datetime
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)

# Database Configuration
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///songs.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SECRET_KEY'] = 'secret_369'
db = SQLAlchemy(app)

login_manager = LoginManager()
login_manager.init_app(app)

from flask_migrate import Migrate

# Initialize Flask-Migrate
migrate = Migrate(app, db)

# Define the User model
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), unique=True, nullable=False)
    password_hash = db.Column(db.String(128), nullable=False)

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

# Define the Survey model to store user preferences
class Response(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    mood = db.Column(db.String(50), nullable=False)
    preferred_mood_songs = db.Column(db.String(200), nullable=False)

class ShareablePlaylist(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    song_id = db.Column(db.Integer, db.ForeignKey('song.id'), nullable=False)
    shareable_link = db.Column(db.String(200), unique=True, nullable=False)

    song = db.relationship('Song', backref=db.backref('playlist_songs', lazy=True))

class PlaylistSong(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    playlist_id = db.Column(db.Integer, db.ForeignKey('shareable_playlist.id'), nullable=False)
    song_id = db.Column(db.Integer, db.ForeignKey('song.id'), nullable=False)

    playlist = db.relationship('ShareablePlaylist', backref=db.backref('songs', lazy=True))
    song = db.relationship('Song')


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
    
    if response.status_code == 200:
        results = response.json().get('items')
        if results:
            video_id = results[0]['id']['videoId']
            youtube_link = f"https://www.youtube.com/watch?v={video_id}"
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


def recommend_song(mood):
    print(f"Received mood for recommendation: {mood}")
    mood = mood.lower()

    # Fetch the survey response for the given mood
    user_survey = Response.query.filter_by(mood=mood).first()

    if user_survey:
        # Use the preferred mood songs from the survey response
        preferred_mood_songs = user_survey.preferred_mood_songs
        filtered_songs = Song.query.filter(Song.mood == preferred_mood_songs).all()
    else:
        # Default to songs of the same mood if no survey response exists
        filtered_songs = Song.query.filter(Song.mood == mood).all()

    print(f"Filtered songs based on mood: {filtered_songs}")

    if not filtered_songs:
        return 'No songs found for this mood.'

    # Randomly select a song
    song = random.choice(filtered_songs)

    # Get the YouTube link for the song
    youtube_link = get_youtube_link(song.name, song.artist)

    # Save the recommendation to history (global history for all users)
    recommendation = RecommendationHistory(
        song_id=song.id,
        recommended_at=datetime.now()
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
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

from sqlalchemy.sql import func

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

        if username == 'admin' and password == 'adminadmin1234':
            # login_user(UserSession(1))  # Assuming '1' as admin user id
            return redirect('/admin')

        return 'Invalid username or password.'

    return render_template('admin_login.html')

@app.route('/admin', methods=['GET', 'POST'])
# @login_required
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

# CLI command to create an admin user (Optional for future use)
@app.cli.command('create_admin')
def create_admin():
    username = input('Enter admin username: ')
    password = input('Enter admin password: ')
    hashed_password = generate_password_hash(password)
    admin = User(username=username, password_hash=hashed_password)
    db.session.add(admin)
    db.session.commit()
    print('Admin user created.')

@app.route('/survey', methods=['GET', 'POST'])
def survey():
    if request.method == 'POST':
        happy_preference = request.form.get('happy_preference')
        sad_preference = request.form.get('sad_preference')

        # Update or create survey responses
        happy_survey = Response.query.filter_by(mood='happy').first()
        if happy_survey:
            happy_survey.preferred_mood_songs = happy_preference
        else:
            happy_survey = Response(mood='happy', preferred_mood_songs=happy_preference)
            db.session.add(happy_survey)

        sad_survey = Response.query.filter_by(mood='sad').first()
        if sad_survey:
            sad_survey.preferred_mood_songs = sad_preference
        else:
            sad_survey = Response(mood='sad', preferred_mood_songs=sad_preference)
            db.session.add(sad_survey)

        db.session.commit()

        return redirect('/')

    return render_template('survey.html')



if __name__ == '__main__':
    with app.app_context():
        db.create_all()  # This will create the tables in the database
        if Song.query.count() == 0:  # Only load data if the table is empty
            load_csv_data()
    app.run(host = '0.0.0.0', port=4000, debug=True)
