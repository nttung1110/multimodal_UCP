from src.utils import Config
# from .unsupervised_audio_inference import UnsupervisedAudioInference
from .unsupervised_video_inference import UnsupervisedVideoInference
from .unsupervised_audio_inference import UnsupervisedAudioInference

def get_pipeline(config: Config):
    pipelines = {
        'unsupervised_video_inference': UnsupervisedVideoInference,
        'unsupervised_audio_inference': UnsupervisedAudioInference
    }

    return pipelines[config.pipeline.name](config)