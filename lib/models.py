import numpy as np
import cv2
import time
import os

class FaceDetectorWrapper:
    def __init__(self):
        return
    def detect(self, cv2_img): 
        # should return a list of face coordinates
        return
    def detect_with_evaluation(self, cv2_img):
        start = time.time()
        faces = self.detect(cv2_img)
        end = time.time()

        print(f'Interval: {end-start:.2f}s - FPS: {1 / round(end-start, 2):.2f} - Detected: {len(faces)}')
        return faces

class HaarCascade(FaceDetectorWrapper):
    def __init__(self, ):
        super().__init__()
        self.face_cascade = cv2.CascadeClassifier(os.path.join(os.path.dirname(__file__), 'haarcascade_frontalface_default.xml'))
    
    def detect(self, cv2_img):
        gray = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2GRAY)
    
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)

        return [(x, y, x+w, y+h) for (x, y, w, h) in faces]


    

    