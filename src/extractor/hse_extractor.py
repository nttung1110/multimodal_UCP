from src.utils import Config
from hsemotion.facial_emotions import HSEmotionRecognizer

import torch
import numpy as np

class HSEEmotExtractor():
    def __init__(self, config: Config):
        self.config = config

        # init model
        self._init_model()

    def _init_model(self):

        print('=========Initializing HSE Emotion Recognizer Model=========')
        self.model = HSEmotionRecognizer(model_name=self.config.model_name, 
                                         device=self.config.device)
        
        self.softmax = torch.nn.Softmax(dim=1)
        
    def run(self, frame, box):
        [x1, y1, x2, y2] = box
        x1, x2  = min(max(0, x1), frame.shape[1]), min(max(0, x2), frame.shape[1]) # replace with clip?
        y1, y2 = min(max(0, y1), frame.shape[0]), min(max(0, y2), frame.shape[0])
        face_imgs = frame[y1:y2, x1:x2]
        try:
            emotion, scores = self.model.predict_emotions(face_imgs, logits=True)

            scores = self.softmax(torch.Tensor(np.array([scores])))
            es_feature = scores[0].tolist()
            emotion_cat = emotion

            return scores, es_feature, emotion_cat
        except:
            raise RuntimeError
        
