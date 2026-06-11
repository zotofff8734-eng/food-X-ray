from ultralytics import YOLO
from PIL import Image
import os

# --- Configuration ---
# IMPORTANT: You must change this path to point to the '.pt' file 
# generated after you finish training your custom YOLOv8 model.
# This file is usually found in 'runs/detect/train/weights/best.pt'.
MODEL_WEIGHTS_PATH = 'path/to/your/best.pt'

# --- Model Cache ---
_model_cache = {}

def load_yolo_model(model_path: str):
    """Loads and caches the custom YOLOv8 model."""
    if "yolo_model" not in _model_cache:
        print(f"Loading custom YOLOv8 model from: {model_path}")
        if not os.path.exists(model_path):
            print(f"ERROR: Model weights not found at '{model_path}'.")
            print("Please train a model first and update the MODEL_WEIGHTS_PATH in yolo_inference.py")
            return None
        try:
            model = YOLO(model_path)
            _model_cache["yolo_model"] = model
            print("Custom YOLOv8 model loaded successfully.")
        except Exception as e:
            print(f"Error loading YOLOv8 model: {e}")
            return None
    return _model_cache["yolo_model"]

def predict_image(image_path: str) -> list:
    """
    Runs inference on a single image and returns a list of detected dishes.
    
    :param image_path: Path to the image file.
    :return: A list of dictionaries, where each dict contains info about a detected dish.
    """
    model = load_yolo_model(MODEL_WEIGHTS_PATH)
    if model is None:
        return []

    try:
        # Run inference
        results = model(image_path)
        
        detected_items = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Get class name
                class_id = int(box.cls[0])
                class_name = model.names[class_id]
                
                # Get bounding box coordinates
                x1, y1, x2, y2 = box.xyxy[0]
                
                detected_items.append({
                    "name": class_name,
                    "box": [int(x1), int(y1), int(x2), int(y2)],
                    "confidence": float(box.conf[0])
                })
        
        print(f"Detected items: {detected_items}")
        return detected_items

    except Exception as e:
        print(f"An error occurred during prediction: {e}")
        return []

if __name__ == '__main__':
    # A test to run this script directly
    # IMPORTANT: This will fail until you provide a real model path and a real image path.
    print("--- Running YOLOv8 Inference Test ---")
    
    # Create a dummy image for testing if it doesn't exist
    test_image = "test_inference.png"
    if not os.path.exists(test_image):
        Image.new('RGB', (640, 480), 'grey').save(test_image)
        print(f"Created dummy image: {test_image}")

    # This will likely fail because the model path is a placeholder
    print(f"Attempting to run prediction on '{test_image}' with model path '{MODEL_WEIGHTS_PATH}'")
    predictions = predict_image(test_image)
    
    if predictions:
        print("
--- Prediction Results ---")
        for item in predictions:
            print(f"  - Detected: {item['name']} (Confidence: {item['confidence']:.2f}) at box {item['box']}")
    else:
        print("
No items detected or an error occurred. This is expected if the model path is a placeholder.")
