# Unified implementation for multimodal UCP
## Installation

```
conda create --name multimodal_ucp python==3.8
pip install --upgrade pip
pip install -r requirements.txt
pip install fastreid
```

## Overview

The unsupervised approach is mainly divided into three sequential steps:
- Individual tracking, and extracting individual emotion signals
- Unsupervised change point algorithm
- Finding final change point with aggregation technique.

In terms of the flow, the inference pipeline for three different modalities are the same. The only difference lies in detailed implementation of each aforementioned component. Therefore, it is essential to specify what components are used for each type of inference pipeline. Here we have three types of pipeline corresponding to 3 different modalities: video, text, audio


## How to run
All experiments are executed via bash files located in ```scripts```. Note that there are two types of video bash files. It depends on how you want to load the data, whether it is a csv file containing all of the information about video, or a video folder.

Checkpoints (you won't have to download the checkpoints unless you want to experiment these extractors):
- [Default deepsort extractor](https://drive.google.com/file/d/1_qwTWdzT9dWNudpusgKavj_4elGgbkUN/view?usp=sharing)
- [Checkpoint for text modality](https://drive.google.com/file/d/18ROp7W-L1k81-YcZ-0amS8PugLt4B04b/view?usp=sharing)

### Video
In total, we have experimented through 8 different combinations for change point video inference:

- Two types of face tracker (for individual tracking component): deep sort, and iou tracker;
- Two face detector algorithms: retinaface, and mtcnn detector
- Two feature extractors (for extracting emotional signals): hse emotion recognizer, and deep face extractor. The latter is just simply the embedding of face being trained on a large dataset. It could provide some helpful features for the model.

The overall template for running video inference script could be as follows:

```
#!/bin/bash

PYTHONPATH='.':$PYTHONPATH \


python main.py \
--config $aggregator_yml$ \
$ucp_yml$ \
$VideoFolderorCSV_yml$ \
configs/pipeline/video_pipeline/base_pipeline.yml \
$face_detector_yml$ \
$tracker_yml$ \
$emotional_signal_extractor_yml$ \
```

It is required to specify components of inference pipeline: which detector, which tracker, … You could find these yaml files in ```config``` folder where we have already configured to choose good parameters in our experiment. You can also refer to comments we made for each param in config file for further information and change them if you want

A few examples of video inference script that we have prepared could also be found in ```scripts/video_scripts/```

You also need to change the path to the video data in ```configs/data/video.yml``` or ```configs/data/video_csv.yml```

We have experimented through different combinations of component to select the optimal one for usage. You could execute that one as follows or you could choose your own combination following above instructions:
```
bash scripts/video_scripts/inference_on_csv/detector_retinaface-tracker_deepsort-extractor_emotion-inference.sh
```

To do the inference on video, use:
```
bash scripts/video_scripts/inference_on_video/detector_retinaface-tracker_deepsort-extractor_emotion-inference.sh
```