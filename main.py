import cv2
from ultralytics import YOLO
import numpy as np

# Define class labels based on the classes the YOLO model was trained on
class_labels = [
    "person", "bicycle", "car", "motorbike", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
    "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]

cap = cv2.VideoCapture("car.mp4")
model = YOLO("yolov8m.pt")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    results = model(frame, device="mps")
    result = results[0]
    bboxes = np.array(result.boxes.xyxy.cpu(), dtype="int")
    classes = np.array(result.boxes.cls.cpu(), dtype="int")
    
    for cls, bbox in zip(classes, bboxes):
        (x, y, x2, y2) = bbox
        class_name = class_labels[cls]
        cv2.rectangle(frame, (x, y), (x2, y2), (0, 0, 255), 2)  # Changed color to red
        cv2.putText(frame, class_name, (x, y - 5), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)  # Changed color to red
    cv2.imshow("Img", frame)
    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
