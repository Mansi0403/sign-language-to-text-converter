import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load your trained model
model = load_model('asl_model.h5')
print("Model loaded! Press 'q' to quit")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
        
    # Resize to model input size
    frame_resized = cv2.resize(frame, (64, 64))
    frame_gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
    frame_normalized = frame_gray / 255.0
    frame_input = np.expand_dims(frame_normalized, axis=(0, -1))
    
    # Predict
    prediction = model.predict(frame_input)
    predicted_class = np.argmax(prediction)
    confidence = np.max(prediction)
    
    # Simple labels (A-Z + nothing + space)
    labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ nothing space"
    detected_letter = labels[predicted_class]
    
    # Display
    cv2.putText(frame, f"Detected: {detected_letter} ({confidence:.2f})", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.imshow('Sign Language AI', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("AI closed!")
