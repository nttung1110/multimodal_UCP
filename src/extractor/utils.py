from src.utils import Config
from .trill_extractor import TrillExtractor
from .spectrogram_extractor import SpectrogramExtractor
from .mtcnn_detector import MTCNNDetector
from .iou_tracker import IoUTracker
from .hse_extractor import HSEEmotExtractor


def get_extractor(config: Config):
    extractor = {
        # for audio
        'trill': TrillExtractor,
        'spectrogram': SpectrogramExtractor,
        # for video
        'hse': HSEEmotExtractor,
}
    return extractor[config.extractor.name](config.extractor)


def get_detector(config: Config): # face detector
    detector = {
        'mtcnn': MTCNNDetector
    }

    return detector[config.detector.name](config.detector)


def get_tracker(config: Config):
    tracker = {
        'iou': IoUTracker
    }

    return tracker[config.tracker.name](config.tracker)