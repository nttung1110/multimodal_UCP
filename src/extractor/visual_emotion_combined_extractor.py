from src.utils import Config
from hsemotion.facial_emotions import HSEmotionRecognizer
from deepface import DeepFace
import torch
import numpy as np
import pdb

class VisualEmotionCombinedExtractor():
    def __init__(self, config: Config):
        self.config = config

        # init emotion model
        self._init_emotion_model()

    def _init_emotion_model(self):

        print('=========Initializing HSE Emotion Recognizer Model=========')
        self.model = HSEmotionRecognizer(model_name=self.config.emotion_model_name, 
                                         device='cuda')
        
        self.softmax = torch.nn.Softmax(dim=1)
        
    def run(self, frame, box):
        [x1, y1, x2, y2] = box
        
        x1, x2  = min(max(0, x1), frame.shape[1]), min(max(0, x2), frame.shape[1]) # replace with clip?
        y1, y2 = min(max(0, y1), frame.shape[0]), min(max(0, y2), frame.shape[0])
        face_imgs = frame[min(y1, y2):max(y1, y2), min(x1, x2):max(x1, x2)]
        
        # Extract facial visual features
        embedding_objs = DeepFace.represent(face_imgs, model_name = self.config.visual_model_name, enforce_detection = self.config.enforce_detection)
        visual_features = np.array(embedding_objs[0]["embedding"])
       
        emotion, scores = self.model.predict_emotions(face_imgs, logits=True)

        scores = self.softmax(torch.Tensor(np.array([scores])))
        es_feature = scores[0].tolist()
        emotion_cat = emotion
        
        combined_features = np.concatenate([visual_features, np.array(es_feature)])

        return scores, combined_features, emotion_cat
        
