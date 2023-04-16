# Unified implementation for multimodal UCP
## Installation

```
conda create --name multimodal_ucp python==3.8
conda env update -n multimodal_ucp --file environment.yml

# These are installed independently for avoiding conflicts
pip3 install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio===0.11.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html

pip install -r requirements.txt

```

## Overview

The unsupervised approach is mainly divided into three sequential steps:
- Individual tracking, and extracting individual emotion signals
- Unsupervised change point algorithm
- Finding final change point with aggregation technique.

In terms of the flow, the inference pipeline for three different modalities are the same. The only difference lies in detailed implementation of each aforementioned component. Therefore, it is essential to specify what components are used for each type of inference pipeline. Here we have three types of pipeline corresponding to 3 different modalities: video, text, audio


## How to run
All experiments are executed via bash file located in ```scripts```. Note that there are two types of video bash file. It depends on how you want to load the data, whether it is a csv file containing all of the information about video, or a video folder.

### Video
In total, we have experimented through 8 different combinations for change point video inference:

- Two types of face tracker (for individual tracking component): deep sort, and iou tracker;
- Two face detector algorithms: retinaface, and mtcnn detector
- Two feature extractor (for extracting emotional signals): hse emotion recognizer, and deep face extractor. The latter is just simply the embedding of face being trained on a large dataset. It could provide some helpful features for the model.

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

It is required to specify components of inference pipeline: which detector, which tracker, … You could find these yaml files in config folder where we have already configured to choose good parameters in our experiment. You can also refer to comments we made for each param in config file for further information and change them if you want

A few examples of video inference script that we have prepared could also be found in ```scripts/video_scripts/```


