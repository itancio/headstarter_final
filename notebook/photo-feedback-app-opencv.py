import os
from flask import Flask, request, jsonify
import cv2 as opencv
import numpy as np
from PIL import Image
import io
from sklearn.cluster import KMeans

app = Flask(__name__)

def analyze_image_content(image_path):
    # Read image
    image = opencv.imread(image_path)
    
    # Analyze colors
    def get_dominant_colors(image, k=3):
        image = opencv.cvtColor(image, opencv.COLOR_BGR2RGB)
        pixels = image.reshape((-1, 3))
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(pixels)
        colors = kmeans.cluster_centers_
        return [f"#{int(color[0]):02x}{int(color[1]):02x}{int(color[2]):02x}" for color in colors]
    
    dominant_colors = get_dominant_colors(image)
    
    # Analyze objects using OpenCV's built-in object detector
    def detect_objects(image):
        gray = opencv.cvtColor(image, opencv.COLOR_BGR2GRAY)
        detector = opencv.SimpleBlobDetector_create()
        keypoints = detector.detect(gray)
        return len(keypoints)
    
    object_count = detect_objects(image)
    
    # Analyze brightness
    def get_brightness(image):
        hsv = opencv.cvtColor(image, opencv.COLOR_BGR2HSV)
        _, _, v = opencv.split(hsv)
        return np.mean(v)
    
    brightness = get_brightness(image)
    brightness_label = "bright" if brightness > 127 else "dark"
    
    # Analyze edges
    def analyze_edges(image):
        gray = opencv.cvtColor(image, opencv.COLOR_BGR2GRAY)
        edges = opencv.Canny(gray, 100, 200)
        edge_percentage = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
        return "high detail" if edge_percentage > 0.1 else "low detail"
    
    detail_level = analyze_edges(image)
    
    # Detect faces (if any)
    def detect_faces(image):
        gray = opencv.cvtColor(image, opencv.COLOR_BGR2GRAY)
        face_cascade = opencv.CascadeClassifier(opencv.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        return len(faces)
    
    face_count = detect_faces(image)
    
    return {
        "dominant_colors": dominant_colors,
        "object_count": object_count,
        "brightness": brightness_label,
        "detail_level": detail_level,
        "face_count": face_count
    }

def generate_feedback(image_analysis):
    feedback = f"Your image appears to be {image_analysis['brightness']} with {image_analysis['detail_level']}. "
    feedback += f"The dominant colors are {', '.join(image_analysis['dominant_colors'])}. "
    feedback += f"I detected approximately {image_analysis['object_count']} distinct elements in the image. "
    if image_analysis['face_count'] > 0:
        feedback += f"The image contains {image_analysis['face_count']} face(s)."
    return feedback

def generate_design_recommendations(image_analysis):
    recommendations = {
        "color_palette": image_analysis['dominant_colors'],
        "visual_treatment": "Consider adding contrast" if image_analysis['brightness'] == "dark" else "Try softening highlights",
        "composition": "The image seems busy" if image_analysis['object_count'] > 10 else "The composition looks balanced",
        "focus": "Consider simplifying the image" if image_analysis['detail_level'] == "high detail" else "Try adding more points of interest"
    }
    if image_analysis['face_count'] > 0:
        recommendations["portrait_mode"] = "Consider using portrait mode or blurring the background to emphasize the face(s)"
    return recommendations

@app.route('/analyze', methods=['POST'])
def analyze_photo():
    if 'photo' not in request.files:
        return jsonify({"error": "No photo file provided"}), 400
    
    photo = request.files['photo']
    photo_path = "temp_photo.jpg"
    photo.save(photo_path)
    
    # Analyze image
    image_analysis = analyze_image_content(photo_path)
    
    # Generate feedback and recommendations
    feedback = generate_feedback(image_analysis)
    recommendations = generate_design_recommendations(image_analysis)
    
    # Clean up temporary file
    os.remove(photo_path)
    
    return jsonify({
        "feedback": feedback,
        "recommendations": recommendations,
        "analysis": image_analysis
    })

if __name__ == '__main__':
    app.run(debug=True)
