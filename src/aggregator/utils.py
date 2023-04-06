from src.utils import Config

from .base_aggregator import BaseCPAggregator
from .video_aggregator import VideoCPAggregator
from .audio_aggregator import AudioCPAggregator

def get_aggregator(config: Config):
    aggregator = {
        'base': BaseCPAggregator,
        'video': VideoCPAggregator,
        'audio': AudioCPAggregator,
    }
    return aggregator[config.aggregator.name](config.aggregator)   