# YOLOv8 Pothole Segmentation & Detection

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![YOLOv8](https://img.shields.io/badge/YOLO-v8-green.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red.svg)

A complete machine learning pipeline for detecting and segmenting potholes in road images using **YOLOv8**. This repository contains the training scripts, inference scripts, and a web application interface to interact with the models dynamically.

---

## Features
- **YOLOv8 Segmentation**: Utilizes state-of-the-art YOLO architecture to accurately mask out potholes.
- **Local Inference**: Includes headless scripts (`detect.py`) for processing images directly on your machine.
- **Interactive Web App**: A completely functional Streamlit application (`app.py`) for real-time inference via a local web interface.

## Repository Structure
- `train.py`: The pipeline used for fine-tuning YOLOv8 on our localized pothole dataset.
- `detect.py`: Console-based inference script for rapid model validation.
- `app.py`: Streamlit web-based UI for uploading images and evaluating them visually.
- `weight/`: Contains the optimized local weights (`best.pt`) post-training.
- `result/`: Hosts local outputs from headless automated validation loops.

## Installation & Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/Adwaith585/yolov8-potholes-mulearn.git
   cd yolov8-potholes-mulearn
   ```
2. Install the necessary requirements:
   ```bash
   pip install -r requirements.text
   ```

## Usage

### Starting the Web App
To launch the interactive GUI, simply run:
```bash
streamlit run app.py
```

### Running Headless Inference
To run detection directly through the terminal without the web interface:
```bash
python detect.py
```
