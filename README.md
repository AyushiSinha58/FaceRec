🧠 FaceRec – Facial Detection & Recognition System

A Python-based facial recognition system that detects and identifies faces from images, webcam captures, and live video streams using machine learning techniques.

📌 Overview
This project implements a complete face detection and recognition pipeline using the face_recognition library.
It encodes facial features from images and matches them against stored data to identify individuals.

It supports:
- Image-based recognition
- Live camera capture
- Real-time video recognition
- Model validation

The system uses HOG (default) and CNN (optional) models for detection and encoding.

🚀 Features
- Detect faces in images and videos
- Identify known individuals with accuracy percentage
- Train model on custom datasets
- Real-time webcam recognition
- Validation using test dataset
- Command-line interface with multiple modes

🛠️ Tech Stack
- Python
- OpenCV (cv2)
- face_recognition
- dlib
- NumPy
- Pillow (PIL)
- argparse
- pickle

🧪 How It Works
1. Training Phase
- Extracts facial encodings from images
- Stores them in encodings.pkl
2. Recognition Phase
- Detects faces in input
- Compares encodings with trained data
- Outputs name and match accuracy
3. Matching Logic
- Uses encoding comparison and voting mechanism
- Returns most probable identity with percentage score

▶️ Usage
🔹 Train the Model
python detector.py --train
🔹 Validate Model
python detector.py --validate
🔹 Test on Image
python detector.py --test -file "path_to_image"
🔹 Capture Image from Webcam
python detector.py --test -capture
🔹 Live Video Recognition
python detector.py --test -video
(Press Q to exit video mode.)
🔹 Change Detection Mode
-modes hog   # CPU (default)
-modes cnn   # GPU (faster, more accurate)

📸 Sample Output
- Bounding box around detected face
- Name of person
- Accuracy percentage (e.g., Ayushi 88%)
- If no match is found → "Unknown"

📊 Results
i. Successfully recognized multiple faces from dataset
ii. Works with:
-- Stored images
-- Live camera input
-- Video streams
iii. Accuracy depends on:
-- Quality of training images
-- Lighting conditions
-- Number of samples

⚠️ Limitations
- Performance drops in low lighting
- Video processing may lag on CPU
- Accuracy varies with face angles
- Requires sufficient training data

📜 License
This project is for educational purposes.
