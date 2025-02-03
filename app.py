from flask import Flask, request, jsonify, render_template
import cv2
import numpy as np
from deepface import DeepFace

app = Flask(__name__)

# Load the face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

@app.route('/')  # Add this route
def home():
    return render_template('index.html')  # Serve the frontend page

@app.route('/analyze_emotion', methods=['POST'])
def analyze_emotion():
    try:
        # Read image from request
        file = request.files['image']
        np_img = np.frombuffer(file.read(), np.uint8)
        frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

        # Convert frame to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        emotions = []

        for (x, y, w, h) in faces:
            # Extract face ROI
            face_roi = frame[y:y+h, x:x+w]

            # Analyze emotion
            result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)
            dominant_emotion = result[0]['dominant_emotion']

            # Convert numpy int to Python int to make it JSON serializable
            emotions.append({
                "x": int(x), 
                "y": int(y), 
                "w": int(w), 
                "h": int(h), 
                "emotion": dominant_emotion
            })

        return jsonify({"success": True, "emotions": emotions})

    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)