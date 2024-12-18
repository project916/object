from flask import Flask, request, jsonify
import cv2
import numpy as np
import os
from uuid import uuid4

app = Flask(__name__)



# Set a maximum file size for uploads (100MB)
app.config['MAX_CONTENT_LENGTH'] = 1000 * 1024 * 1024  # 100 MB

elephant_found = ''
def load_yolo_model():
        # Load YOLO model configuration and weights
    net = cv2.dnn.readNet("yolov4.weights", "yolov4.cfg")
    
    # Load class names
    with open("coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    
    # Get output layers
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    
    return net, classes, output_layers
    

def backend(video_path):
    detect_elephant_or_bear(video_path)
    if elephant_found == 'elephant':
        print("Elephant Found")
        return 'Elephant Found, '
    elif elephant_found == 'bear':
        print("Bear Found")
        return 'Bear Found'
        
    else:
        return 'notfound'


def detect_elephant_or_bear(video_path):

    try:
        global elephant_found  # Use the global variable
    
        net, classes, output_layers = load_yolo_model()

        # Open video file
        cap = cv2.VideoCapture(video_path)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            height, width, _ = frame.shape

            # Prepare input for YOLO
            blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
            net.setInput(blob)

            # Forward pass
            outs = net.forward(output_layers)

            # Process detections
            for out in outs:
                for detection in out:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]

                    if confidence > 0.5 and classes[class_id] == "elephant":
                        elephant_found = 'elephant'
                        cap.release()
                        return
                    elif confidence > 0.5 and classes[class_id] == "bear":
                        elephant_found = 'bear'
                        cap.release()
                        return
                    else:
                        elephant_found = 'Not Found'
                        cap.release()
        
            
        cap.release()
       

    except Exception as e:
        print(f"[ERROR] Error during detection: {e}")
        return "Error during detection"


@app.route('/upload', methods=['POST'])
def upload_video():
    """
    Endpoint to upload a video and detect objects in it.
    Returns:
        JSON response with the detection result.
    """
    try:
        if 'video' not in request.files:
            print("[ERROR] No video file provided in request.")
            return jsonify({"error": "No video file provided"}), 400

        video_file = request.files['video']
        if video_file.filename == '':
            print("[ERROR] No file name provided.")
            return jsonify({"error": "No file name provided"}), 400

        # Ensure the uploads folder exists
        upload_folder = "uploads"
        os.makedirs(upload_folder, exist_ok=True)  # Create folder if it doesn't exist

        # Save video to uploads folder with a unique name
        unique_filename = f"{uuid4()}_{video_file.filename}"
        video_path = os.path.join(upload_folder, unique_filename)
        video_file.save(video_path)

        if not os.path.exists(video_path):
            print("[ERROR] Video file was not saved.")
            return jsonify({"error": "File upload failed"}), 500

        print(f"[INFO] Video uploaded to {video_path}")

        # Detect elephant or bear in the video
        result = backend(video_path)

        # Optionally delete the uploaded video after processing
        try:
            os.remove(video_path)
            print(f"[INFO] Deleted video file {video_path} after processing.")
        except Exception as cleanup_error:
            print(f"[WARNING] Could not delete video file {video_path}: {cleanup_error}")

        return jsonify({"result": result})

    except Exception as e:
        print(f"[ERROR] Error in upload_video: {e}")
        return jsonify({"error": "An error occurred during processing"}), 500


if __name__ == '__main__':
    app.run(debug=True, port=8080)
