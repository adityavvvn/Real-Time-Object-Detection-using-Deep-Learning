import cv2
import numpy as np

# Load YOLO model
def load_yolo():
    net = cv2.dnn.readNet("yolov4.weights", "yolov4.cfg")
    with open("coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    return net, classes, output_layers

# Draw bounding boxes
def draw_bounding_boxes(boxes, confidences, class_ids, classes, frame):
    for i in range(len(boxes)):
        x, y, w, h = boxes[i]
        label = f"{classes[class_ids[i]]}: {confidences[i]:.2f}"
        color = (0, 255, 0)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# Detect objects
def detect_objects(net, output_layers, frame, classes):
    height, width, channels = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    boxes = []
    confidences = []
    class_ids = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.3:  # Lower threshold
                center_x, center_y, w, h = (detection[0:4] * np.array([width, height, width, height])).astype(int)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    if len(indexes) > 0:
        indexes = indexes.flatten()
        boxes = [boxes[i] for i in indexes]
        confidences = [confidences[i] for i in indexes]
        class_ids = [class_ids[i] for i in indexes]
    else:
        boxes, confidences, class_ids = [], [], []

    return boxes, confidences, class_ids

# List available cameras
def list_cameras():
    print("Detecting available cameras...")
    available_cameras = []
    for i in range(10):  # Check up to 10 cameras
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available_cameras.append(i)
            cap.release()
    return available_cameras

# Choose camera
def choose_camera():
    cameras = list_cameras()
    if not cameras:
        print("No cameras detected!")
        exit()
    
    print("Available cameras:")
    for cam in cameras:
        print(f"Camera {cam}")
    
    selected_cam = int(input("Select a camera (number): "))
    if selected_cam not in cameras:
        print("Invalid selection!")
        exit()
    
    return selected_cam

# Real-time detection
def real_time_detection():
    net, classes, output_layers = load_yolo()
    selected_camera = choose_camera()
    cap = cv2.VideoCapture(selected_camera)  # Open selected camera

    if not cap.isOpened():
        print(f"Error: Could not open camera {selected_camera}.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        boxes, confidences, class_ids = detect_objects(net, output_layers, frame, classes)
        draw_bounding_boxes(boxes, confidences, class_ids, classes, frame)
        cv2.imshow("Real-Time Object Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):  # Exit on pressing 'q'
            break

    cap.release()
    cv2.destroyAllWindows()

# Run
if __name__ == "__main__":
    real_time_detection()
