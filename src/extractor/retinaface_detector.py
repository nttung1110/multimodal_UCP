from src.utils import Config
from deepface import DeepFace
import numpy as np
import pdb

class RetinaFaceDetector():
    def __init__(self, config: Config):
        self.config = config
        
    def run(self, frame):
        face_objs = DeepFace.extract_faces(frame, detector_backend=self.config.detector_backend, enforce_detection = self.config.enforce_detection)
        bounding_boxes = []
        probs = []
        for face_obj in face_objs:
            x,y,w,h = face_obj['facial_area']['x'], face_obj['facial_area']['y'], \
                        face_obj['facial_area']['w'], face_obj['facial_area']['h']
            bounding_boxes.append([x,y,w,h])
            probs.append(face_obj['confidence'])
            
        return np.array(bounding_boxes), np.array(probs)