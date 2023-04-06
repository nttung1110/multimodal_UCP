from src.utils.config import Config

from .base_folder import BasefolderLoader
# from .audio_folder import  AudioFolder
from .video_folder import VideofolderLoader

def get_loader(config: Config):
    loader = {
        'base_folder': BasefolderLoader,
        'video_folder': VideofolderLoader
    }

    return loader[config.data.name](config)
