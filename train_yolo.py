from ultralytics import YOLO
import torch
import os

def train_food_model():
    """
    This script trains a YOLOv8 model on a custom food dataset.
    
    Before running, make sure you have:
    1. A prepared dataset in YOLO format.
    2. A `data.yaml` file pointing to your train/validation sets and listing class names.
    """
    
    # --- 1. CONFIGURATION ---
    
    # Path to your dataset configuration file.
    # This path now points to the default 'data.yaml' in the created 'dataset' directory.
    dataset_yaml_path = 'dataset/data.yaml' 
    
    # Number of training epochs. 100 is a good start, more may be needed.
    num_epochs = 100
    
    # Image size for training. 640 is common for YOLOv8.
    image_size = 640
    
    # Choose a pretrained model to start from.
    # 'yolov8n.pt' is the smallest and fastest ('n' for nano).
    # For more accuracy, you can use 'yolov8s.pt' (small), 'yolov8m.pt' (medium), etc.
    pretrained_model = 'yolov8n.pt'

    # --- 2. MODEL TRAINING ---
    
    print("--- YOLOv8 Custom Training ---")
    
    # Safety check for the dataset path
    if not os.path.exists(os.path.dirname(dataset_yaml_path)) and dataset_yaml_path != 'path/to/your/dataset/data.yaml':
        print("
ERROR: The directory for your dataset does not seem to exist.")
        print("Please update the 'dataset_yaml_path' variable in this script with the correct path.")
        print("Expected something like: 'C:/path/to/dataset/data.yaml'")
        return

    print(f"Using pretrained model: {pretrained_model}")
    print(f"Dataset config: {dataset_yaml_path}")
    print(f"Number of epochs: {num_epochs}")
    print(f"Image size: {image_size}")
    
    # Check for GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Training on device: {device}")
    if device == 'cpu':
        print("WARNING: No GPU detected. Training on CPU will be very slow.")

    try:
        # Load a pretrained YOLOv8 model
        model = YOLO(pretrained_model)

        # Train the model on your custom dataset
        print("
Starting training... (This may take a long time)")
        results = model.train(
            data=dataset_yaml_path,
            epochs=num_epochs,
            imgsz=image_size,
            device=device,
            patience=20 # Stop training if no improvement after 20 epochs
        )
        
        print("
--- Training Complete ---")
        print("Model and training results are saved in the 'runs/detect/train' directory.")
        print("Your final model is likely at 'runs/detect/train/weights/best.pt'.")
        print("You can now use this 'best.pt' file for inference in the bot.")

    except FileNotFoundError:
        print(f"
ERROR: Dataset configuration file not found at '{dataset_yaml_path}'.")
        print("Please make sure the path is correct.")
    except Exception as e:
        print(f"
An error occurred during training: {e}")


if __name__ == '__main__':
    train_food_model()
