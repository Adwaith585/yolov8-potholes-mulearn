from ultralytics import YOLO
import os

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    model_path = os.path.join("runs", "pothole_seg_model", "weights", "best.pt")
    source_dir = os.path.join("dataset", "valid", "images")
    runs_dir = "runs"
    
    if not os.path.exists(model_path):
        print(f"Model path {model_path} not found. Please run train.py first.")
        return

    print("Loading the trained model...")
    model = YOLO(model_path)
    
    print(f"Running detection on images in {source_dir}...")
    results = model.predict(
        source=source_dir,
        save=True,
        project=runs_dir,
        name="pothole_seg_predict"
    )
    
    print("Detection complete!")

if __name__ == "__main__":
    main()
