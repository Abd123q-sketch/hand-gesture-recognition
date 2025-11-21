import cv2
import mediapipe as mp # type: ignore
import numpy as np

print("OpenCV version :", cv2.__version__)
print("Mediapipe imported successfully!")
print("Numpy version :", np.__version__)

# Test caméra
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ La caméra ne s'est pas ouverte !")
else:
    print("✅ Caméra détectée avec succès !")
    cap.release()
