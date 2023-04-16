from src.utils import Config
from facenet_pytorch import MTCNN

class MTCNNDetector():
    def __init__(self, config: Config):
        self.config = config

        # init model
        self._init_model()

    def _init_model(self):
        print('=========Initializing MTCNN Face Detector Model=========')
        self.model = MTCNN(keep_all=self.config.keep_all, 
                           post_process=self.config.post_process, 
                           min_face_size=self.config.min_face_size, 
                           device='cuda:0')
        
    def run(self, frame):
        bounding_boxes, probs = self.model.detect(frame, landmarks=False)
        
        return bounding_boxes, probs