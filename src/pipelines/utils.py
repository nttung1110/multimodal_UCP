from src.utils import Config
# from .unsupervised_audio_inference import UnsupervisedAudioInference
from .unsupervised_video_inference import UnsupervisedVideoInference
from .unsupeervised_audio_inference import UnsupervisedAudioInference
from .unsupervised_audio_inference_v1 import UnsupervisedAudioJointInference
from .unsupervised_text_inference import UnsupervisedTextInference

def get_pipeline(config: Config):
    pipelines = {
        'unsupervised_video_inference': UnsupervisedVideoInference,
        'unsupervised_audio_inference': UnsupervisedAudioInference,
        'unsupervised_text_inference': UnsupervisedTextInference,
        'unsupervised_audio_inference_v1': UnsupervisedAudioJointInference
    }

    return pipelines[config.pipeline.name](config)