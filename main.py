# Import required libraries
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
from flask import Flask, Response, render_template
import threading
from pathlib import Path
import time
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F
import json

class ObjectDetectionSystem:
    def __init__(self, model_type="FRCNN", confidence_threshold=0.5):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.confidence_threshold = confidence_threshold
        self.model = self._load_model(model_type)
        self.classes = self._load_coco_classes()
        
    def _load_model(self, model_type):
        """Load and configure the detection model"""
        if model_type == "FRCNN":
            model = fasterrcnn_resnet50_fpn(pretrained=True)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
            
        model.eval()
        model.to(self.device)
        return model
    
    def _load_coco_classes(self):
        """Load COCO dataset class names"""
        coco_names = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
            'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
            'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
            'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
            'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
            'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
            'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
            'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
            'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
            'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
            'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
            'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
            'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]
        return coco_names

    def preprocess_image(self, image):
        """Preprocess image for model input"""
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Convert to tensor
        image_tensor = F.to_tensor(image)
        
        # Add batch dimension
        image_tensor = image_tensor.unsqueeze(0)
        
        return image_tensor.to(self.device)

    def detect_objects(self, image):
        """Perform object detection on an image"""
        # Preprocess image
        image_tensor = self.preprocess_image(image)
        
        # Get predictions
        with torch.no_grad():
            predictions = self.model(image_tensor)
        
        # Process predictions
        boxes = predictions[0]['boxes'].cpu().numpy()
        scores = predictions[0]['scores'].cpu().numpy()
        labels = predictions[0]['labels'].cpu().numpy()
        
        # Filter by confidence threshold
        mask = scores >= self.confidence_threshold
        boxes = boxes[mask]
        scores = scores[mask]
        labels = labels[mask]
        
        return boxes, scores, labels

    def draw_predictions(self, image, boxes, scores, labels):
        """Draw bounding boxes and labels on image"""
        for box, score, label in zip(boxes, scores, labels):
            # Convert box coordinates to integers
            box = box.astype(int)
            
            # Draw bounding box
            cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
            
            # Add label and score
            label_text = f"{self.classes[label-1]}: {score:.2f}"
            cv2.putText(image, label_text, (box[0], box[1]-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return image

class VideoCamera:
    def __init__(self, source=0):
        self.video = cv2.VideoCapture(source)
        self.detector = ObjectDetectionSystem()
        
    def __del__(self):
        self.video.release()
        
    def get_frame(self):
        """Get frame from camera and process it"""
        success, frame = self.video.read()
        if not success:
            return None
            
        # Perform detection
        boxes, scores, labels = self.detector.detect_objects(frame)
        
        # Draw predictions
        processed_frame = self.detector.draw_predictions(frame, boxes, scores, labels)
        
        # Convert frame to jpg
        ret, jpeg = cv2.imencode('.jpg', processed_frame)
        return jpeg.tobytes()

# Flask application
app = Flask(__name__)
camera = None

def gen_frames():
    """Generate camera frames"""
    global camera
    while True:
        frame = camera.get_frame()
        if frame is not None:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/')
def index():
    """Video streaming home page"""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# HTML template (save as templates/index.html)
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Real-Time Object Detection</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background: #f0f0f0;
        }
        .container {
            max-width: 800px;
            margin: 0 auto;
            text-align: center;
        }
        h1 {
            color: #333;
        }
        .video-container {
            margin-top: 20px;
            background: white;
            padding: 10px;
            border-radius: 5px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Real-Time Object Detection</h1>
        <div class="video-container">
            <img src="{{ url_for('video_feed') }}" width="100%">
        </div>
    </div>
</body>
</html>
"""

def create_template_file():
    """Create template file for Flask application"""
    template_dir = Path("templates")
    template_dir.mkdir(exist_ok=True)
    
    with open(template_dir / "index.html", "w") as f:
        f.write(HTML_TEMPLATE)

def main():
    """Main function to run the application"""
    global camera
    
    # Create template file
    create_template_file()
    
    # Initialize camera
    camera = VideoCamera()
    
    # Run Flask app
    app.run(host='0.0.0.0', port=5000, debug=False)

if __name__ == "__main__":
    main()