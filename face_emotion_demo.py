import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
import tensorflow as tf

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

# Define emotion labels
emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Create and compile the model
def create_model():
    model = Sequential([
        Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)),
        Conv2D(64, kernel_size=(3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        Conv2D(128, kernel_size=(3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(128, kernel_size=(3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        Flatten(),
        Dense(1024, activation='relu'),
        Dropout(0.5),
        Dense(7, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def process_frame(frame, face_detection, model):
    # Convert the BGR image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_h, frame_w, _ = frame.shape
    
    # Process the frame and detect faces
    results = face_detection.process(rgb_frame)
    
    # If faces are detected
    if results.detections:
        for detection in results.detections:
            # Get bounding box coordinates
            bbox = detection.location_data.relative_bounding_box
            x = int(bbox.xmin * frame_w)
            y = int(bbox.ymin * frame_h)
            w = int(bbox.width * frame_w)
            h = int(bbox.height * frame_h)
            
            # Ensure coordinates are within frame bounds
            x = max(0, x)
            y = max(0, y)
            w = min(w, frame_w - x)
            h = min(h, frame_h - y)
            
            # Extract face ROI
            face_roi = frame[y:y+h, x:x+w]
            if face_roi.size == 0:
                continue
                
            # Preprocess for emotion classification
            try:
                gray_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
                resized_roi = cv2.resize(gray_roi, (48, 48))
                normalized_roi = resized_roi / 255.0
                input_roi = normalized_roi.reshape(1, 48, 48, 1)
                
                # Predict emotion
                emotion_predictions = model.predict(input_roi, verbose=0)
                emotion_label = emotions[np.argmax(emotion_predictions[0])]
                emotion_confidence = np.max(emotion_predictions[0])
                
                # Draw bounding box and emotion label
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                label = f"{emotion_label}: {emotion_confidence:.2f}"
                cv2.putText(frame, label, (x, y-10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            except Exception as e:
                print(f"Error processing face ROI: {e}")
                continue
    
    return frame

def main():
    # Create and compile the model
    model = create_model()
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    print("Starting real-time emotion detection... Press 'q' to quit")
    
    with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame")
                break
            
            # Process frame
            output_frame = process_frame(frame, face_detection, model)
            
            # Display the result
            cv2.imshow('Real-time Emotion Detection', output_frame)
            
            # Break loop on 'q' press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 