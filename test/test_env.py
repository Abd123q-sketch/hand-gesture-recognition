
import cv2
import mediapipe as mp
import numpy as np

print("OpenCV OK :", cv2.__version__)
print("Mediapipe OK :", mp.__version__)
print("Numpy OK :", np.__version__)

cam = cv2.VideoCapture(0)
print("Camera OK :", cam.isOpened())
cam.release()