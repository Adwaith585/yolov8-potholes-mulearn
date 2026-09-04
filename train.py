from ultralytics import YOLO
import os

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    print("Loading YOLOv8 segmentation model...")
    model = YOLO("yolov8n-seg.pt")

    print("Starting training on pothole dataset...")
    results = model.train(
        data=os.path.join("dataset", "data.yaml"),
        epochs=5,
        imgsz=640,
        project="runs",
        name="pothole_seg_model"
    )

    print("Training complete!")

if __name__ == "__main__":
    main()
