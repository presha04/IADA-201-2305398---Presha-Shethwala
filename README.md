# Traffic Violation Detection Project 



**Project Overview -**
The Traffic Violation Detection Project is an end-to-end solution designed to automatically identify and classify traffic violations from video footage. Utilizing computer vision techniques and deep learning models, this project processes raw traffic videos to detect violations such as red-light running, illegal U-turns, no-entry breaches, and lane deviations. By automating traffic monitoring, this system aims to enhance road safety and assist law enforcement agencies in efficient traffic management.



**Features -** 
- **Dataset Preparation:** Automatically splits a master dataset of traffic videos into training and validation sets.
- **Frame Extraction:** Extracts frames from videos at specified intervals for analysis.
- **Label Generation:** Generates labels for each extracted frame, compatible with YOLO format.
- **Exploratory Data Analysis (EDA):** Provides insights into the distribution and characteristics of traffic violations in the dataset.
- **Model Training:** Leverages YOLOv5 for object detection and violation classification.
- **Violation Detection:** Processes videos to detect and annotate traffic violations in real-time.
- **Evaluation Metrics:** Calculates accuracy and other performance metrics to assess model effectiveness.



**Technologies Used -** 
- **Programming Language:** Python
- **Libraries & Frameworks:**
  - OpenCV for video processing and frame extraction
  - TensorFlow for deep learning tasks
  - NumPy for numerical operations
  - Pandas for data manipulation
  - Matplotlib & Seaborn for data visualization
  - YOLOv5 for object detection
  - PyTorch as the backend for YOLOv5
  - Scikit-learn for evaluation metrics



**Installation -** 

**1. Clone the Repository**
  ```
git clone https://github.com/yourusername/traffic-violation-detection.git
cd traffic-violation-detection
```
**2. Install Required Dependencies**
```
pip install -r requirements.txt
```
**3. Install YOLOv5**
```
git clone https://github.com/ultralytics/yolov5.git
cd yolov5
pip install -r requirements.txt
cd ..
```

**Dataset -** 
- **Master Dataset:** Place all your traffic-related video files in the data/TrafficDataset/ directory. Supported video formats include .mp4, .webm, and .mov.
- **Directory Structure:**
```
data/
└── TrafficDataset/
    ├── videos/
    │   ├── train/
    │   └── val/
    ├── images/
    │   ├── train/
    │   └── val/
    └── labels/
        ├── train/
        └── val/
```

**Data Preparation -** 
- **Dataset Splitting:** Randomly selects 10-12 videos from the master dataset and splits them into training (80%) and validation (20%) sets.
- **Frame Extraction & Label Generation:** Extracts frames from each video at specified intervals (e.g., every 30 frames) and generates corresponding labels in YOLO format.
- **Synthetic Data Generation:** Creates a synthetic dataset for demonstration purposes, including frames, violation statuses, types, and timestamps.
- **Exploratory Data Analysis (EDA):** Analyzes the distribution of violations and their types to understand dataset characteristics.

**Model Training -** 

**1. Navigate to YOLOv5 Directory**
```
cd yolov5
```
**2. Prepare Dataset Configuration**

Ensure that your dataset.yaml file correctly points to the training and validation image and label directories.
```
# dataset.yaml
train: ../data/TrafficDataset/images/train
val: ../data/TrafficDataset/images/val

nc: 5  # Number of classes
names: ['red-light', 'illegal U-turn', 'no-entry', 'lane deviation', 'none']
```
**3. Run Training Script**

Execute the training command from the command line. Ensure you're in the YOLOv5 directory.
```
python train.py --img 640 --batch 10 --epochs 10 --data ../data/TrafficDataset/yolov5/dataset.yaml --weights yolov5s.pt
```
--img: Image size

--batch: Batch size

--epochs: Number of training epochs

--data: Path to dataset configuration

--weights: Pre-trained weights to start with


**4. Monitor Training Progress**

Training logs and metrics will be displayed in the console. Upon completion, the trained model will be saved in the runs/train/ directory.



**Violation Detection -** 

The violation detection module processes videos to identify and annotate traffic violations in real-time. Here's how it works:

**1. Load YOLOv5 Model**
```
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
```
**2. Process Videos**

The detect_violations function reads input videos frame by frame, applies the YOLOv5 model to detect objects, and identifies violations based on detected objects like cars and motorbikes.
```
detect_violations(input_video_path, output_video_path)
```
**3. Output**
- **Annotated Video:** The output video will have bounding boxes around detected objects with labels and confidence scores. If a violation is detected, a "Violation Detected!" message is displayed; otherwise, "No Violation Detected!" is shown.
- **Console Logs:** Information about the number of frames processed and violations detected.

**Evaluation -** 

To assess the performance of the violation detection model, accuracy and other metrics are calculated using ground truth labels and model predictions.

**1. Calculate Accuracy**
```
from sklearn.metrics import accuracy_score

def calculate_accuracy(y_true, y_pred):
    if not y_true or not y_pred:
        print("Error: y_true or y_pred is empty. Cannot calculate accuracy.")
        return None
    accuracy = accuracy_score(y_true, y_pred)
    return accuracy * 100
```
**2. Example Usage**
```
y_true = [0, 1, 0, 1, 0, 1]  # Ground truth labels
y_pred = [0, 1, 0, 0, 0, 1]  # Predicted labels

accuracy = calculate_accuracy(y_true, y_pred)
if accuracy is not None:
    print(f"Accuracy: {accuracy:.2f}%")
```
**3. Extend Evaluation Metrics**
For a comprehensive evaluation, consider incorporating precision, recall, F1-score, and confusion matrices.

- **Refrences:**

https://github.com/anmspro/Traffic-Signal-Violation-Detection-System

https://github.com/MANASgfx/Traffic-Violation-Detection-System

https://www.kaggle.com/datasets/yusufberksardoan/traffic-detection-project/code

https://www.kaggle.com/datasets/meliodassourav/traffic-violation-dataset-v3

https://chatgpt.com/






- *LINK OF THE DATA - https://drive.google.com/drive/u/0/folders/1JkO9M4TyA48eFnUrEHBYfyktMRpZ47d5*
