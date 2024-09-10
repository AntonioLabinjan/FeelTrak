from flask import Flask, render_template, request, jsonify
from deepface import DeepFace
import cv2
import numpy as np
import base64

app = Flask(__name__)

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
        
        return jsonify({'emotion': dominant_emotion}), 200
    
    except Exception as e:
        # Log the error details to the console
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500



if __name__ == '__main__':
    app.run(debug=True)
