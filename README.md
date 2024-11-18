## Real-Time Object Detection System
## Project Overview

This project implements a real-time object detection system using the Faster R-CNN model and the Flask web framework. It enables users to detect objects in real time through a web-based interface powered by a pre-trained model on the COCO dataset.

## Features
Real-time object detection from webcam feeds.
Intuitive web-based user interface for easy interaction.
Pre-trained model leveraging the COCO dataset, supporting 80 object classes.
Adjustable detection parameters (e.g., confidence threshold).

## Installation
1. Clone the Repository:

git clone [your-repository-link]  
cd [repository-folder]  

2. Install Requirements:

pip install -r requirements.txt  

## Usage
1. Run the Application:

python app.py  

2. Access the Web Interface:
Open your web browser and navigate to:
- http://localhost:5000

## Technologies Used
- Python 3.x: Programming language.
- PyTorch: Deep learning framework for model implementation.
- OpenCV: Real-time computer vision library.
- Flask: Micro web framework for the web interface.
- TorchVision: Pre-trained models and vision utilities.
  
## Performance
Model: Faster R-CNN with ResNet50 backbone.
Average Inference Time: [Your measured time] ms per frame.
Confidence Threshold: 0.5 (default).
