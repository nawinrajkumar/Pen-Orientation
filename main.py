import cv2
import time
import numpy as np
from ultralytics import YOLO
from paddleocr import PaddleOCR, draw_ocr
import mediapipe as mp
from scipy.spatial.transform import Rotation as R

# Load YOLOv8 model
model = YOLO('models/pen.pt')  # Replace with your trained model path if necessary

# Initialize PaddleOCR
ocr = PaddleOCR(use_angle_cls=True, lang='en')

# List of class names (make sure the order matches your model's class indices)
class_names = ['notebook', 'pen', 'tip']  # Update this list according to your classes

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False, 
    max_num_hands=1, 
    min_detection_confidence=0.5)

# Function to detect objects
def detect_objects(image):
    results = model(image)
    return results
    
# Function to crop the notebook from the frame
def crop_notebook(frame, results):
    for box in results[0].boxes:
        cls_index = int(box.cls.item())
        cls_name = class_names[cls_index]

        if cls_name == 'notebook':
            x1, y1, x2, y2 = map(
                int, 
                box.xyxy.cpu().numpy().flatten()
                )

            return frame[y1:y2, x1:x2]
    return None

# Function to perform OCR on the cropped image
def perform_ocr(cropped_image):
    result = ocr.ocr(cropped_image, cls=True)
    if result and len(result) > 0 and result[0] is not None:
        text = [line[-1][0] for line in result[0] if line[-1]]
        return '\n'.join(text)
    return ""

# Function to calculate characters per minute
def calculate_cpm(total_start_time, char_count):
    elapsed_time = time.time() - total_start_time
    print(elapsed_time)
    print(char_count)
    if elapsed_time > 0:
        cpm = (char_count / elapsed_time) * 60  # characters per minute
        return cpm
    return 0

# Function to estimate 3D orientation of the pen
def estimate_orientation(frame, results):
    object_points = np.array([
        [0, 0, 0],   # Tip of the pen
        [0, 1, 0],   # End of the pen
        [0, 0, 1],   # Side point
        [1, 0, 0],   # Another side point
        [0.5, 0.5, 0],  # Middle point on the pen
        [-0.5, 0.5, 0]  # Another point on the pen
    ], dtype=np.float32)

    for box in results[0].boxes:
        cls_index = int(box.cls.item())
        cls_name = class_names[cls_index]
        if cls_name == 'pen':
            x1, y1, x2, y2 = map(
                int, 
                box.xyxy.cpu().numpy().flatten())
            image_points = np.array([
                [(x1 + x2) / 2, y1],  # Midpoint top
                [(x1 + x2) / 2, y2],  # Midpoint bottom
                [x1, (y1 + y2) / 2],  # Midpoint left
                [x2, (y1 + y2) / 2],  # Midpoint right
                [(x1 + x2) / 2, (y1 + y2) / 2],  # Center point
                [(x1 + x2) / 2, y1 + (y2 - y1) * 0.75]  # A point 75% down the pen length
            ], dtype=np.float32)

            camera_matrix = np.array([
                [frame.shape[1], 0, frame.shape[1] / 2],
                [0, frame.shape[1], frame.shape[0] / 2],
                [0, 0, 1]
            ], dtype=np.float32)

            dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion

            success, rotation_vector, translation_vector = cv2.solvePnP(
                object_points, image_points, camera_matrix, dist_coeffs)

            if success:
                rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
                r = R.from_matrix(rotation_matrix)
                roll, pitch, yaw = r.as_euler('xyz', degrees=True)
                return roll, pitch, yaw
    return None, None, None

# Function to detect the index finger using MediaPipe
def detect_index_finger(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(frame_rgb)
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            h, w, _ = frame.shape
            x, y = int(index_finger_tip.x * w), int(index_finger_tip.y * h)
            return (x, y)
    return None

# Main loop to process video frames
def main():
    cap = cv2.VideoCapture("input/input.mp4")  # Open the video file

    total_start_time = time.time()
    char_count = 0
    detected_text_history = ""

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Resize the frame for display
        resized_frame = cv2.resize(frame, (640, 480))  # Adjust size as needed

        # Detect objects
        results = detect_objects(resized_frame)

        # Detect the index finger
        index_finger_pos = detect_index_finger(resized_frame)
        if index_finger_pos:
            cv2.circle(resized_frame, index_finger_pos, 5, (0, 255, 0), -1)

        # Crop the notebook from the resized_frame
        cropped_notebook = crop_notebook(resized_frame, results)

        # If notebook is detected and cropped, perform OCR
        if cropped_notebook is not None:
            text = perform_ocr(cropped_notebook)
            if text is not None:
                char_count = len(text.replace(' ', '').replace('\n', ''))

                # Calculate CPM
                cpm = calculate_cpm(total_start_time, char_count)
                cpm_text = f"CPM: {cpm:.2f}"
                print(cpm_text)

            # Display the cropped notebook and OCR result
            for i, line in enumerate(text.split('\n')):
                y = 50 + i * 20
                cv2.putText(resized_frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

        # Estimate the orientation of the pen
        roll, pitch, yaw = estimate_orientation(resized_frame, results)
        if roll is not None and pitch is not None and yaw is not None:
            orientation_text = f"Roll: {roll:.2f}, Pitch: {pitch:.2f}, Yaw: {yaw:.2f}"
            print(orientation_text)
            cv2.putText(resized_frame, orientation_text, (10, resized_frame.shape[0] - 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)



        # Display CPM on the resized_frame
        cv2.putText(resized_frame, cpm_text, (10, resized_frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow('Frame', resized_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
