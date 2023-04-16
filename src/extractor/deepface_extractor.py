from src.utils import Config
from deepface import DeepFace
import torch
import numpy as np
import pdb
        
class DeepFaceExtractor(object):
    def __init__(self, config: Config):
        self.config = config

    def run(self, frame, box):
        [x1, y1, x2, y2] = box
        x1, x2  = min(max(0, x1), frame.shape[1]), min(max(0, x2), frame.shape[1]) # replace with clip?
        y1, y2 = min(max(0, y1), frame.shape[0]), min(max(0, y2), frame.shape[0])
        face_imgs = frame[min(y1, y2):max(y1, y2), min(x1, x2):max(x1, x2)]
        
        embedding_objs = DeepFace.represent(face_imgs, model_name = self.config.model_name, enforce_detection = self.config.enforce_detection)
        features = np.array(embedding_objs[0]["embedding"])
        
        score = 0
        emotion_cat = "None"
        return score, features, emotion_cat