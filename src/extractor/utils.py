from src.utils import Config
from .trill_extractor import TrillExtractor
from .spectrogram_extractor import SpectrogramExtractor
from .mtcnn_detector import MTCNNDetector
from .retinaface_detector import RetinaFaceDetector
from .hse_extractor import HSEEmotExtractor
from .deepface_extractor import DeepFaceExtractor
from .visual_emotion_combined_extractor import VisualEmotionCombinedExtractor
from .iou_tracker import IoUTracker
from .deep_sort_tracker import DeepSortTracker
from .compm_extractor import CoMPMExtractor


def get_extractor(config: Config):
    extractor = {
        # for audio
        'trill': TrillExtractor,
        'spectrogram': SpectrogramExtractor,
        # for video
        'hse': HSEEmotExtractor,
        'deepface': DeepFaceExtractor,
        'visual_emotion_combined': VisualEmotionCombinedExtractor, 
        'compm': CoMPMExtractor, 
}
    return extractor[config.extractor.name](config.extractor)


def get_detector(config: Config): # face detector
    detector = {
        'mtcnn': MTCNNDetector,
        'retinaface': RetinaFaceDetector
    }

    return detector[config.detector.name](config.detector)


def get_tracker(config: Config):
    tracker = {
        'iou': IoUTracker,
        'deepsort': DeepSortTracker
    }

    return tracker[config.tracker.name](config.tracker)