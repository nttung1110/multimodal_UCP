import os
import os.path as osp
import pdb

from pydub import AudioSegment
from tqdm import tqdm

path_mp3 = '../../EVAL_LDC2023E07/PREPROCESSED_AUDIO/mp3_files/'
path_wav = '../../EVAL_LDC2023E07/PREPROCESSED_AUDIO/wav_files/'

for audio_file in tqdm(os.listdir(path_mp3)):
    path_audio_file = osp.join(path_mp3, audio_file)
    path_wav_file = osp.join(path_wav, audio_file.split('.')[0]+'.wav')

    command_run = f'ffmpeg -i {path_audio_file} -vn {path_wav_file}'
    
    # sound = AudioSegment.from_file(path_audio_file, format="mp3")
    # # sound.export(path_wav_file, format="wav")

    os.system(command_run)

