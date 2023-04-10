#!/bin/bash

PYTHONPATH='.':$PYTHONPATH \


python main.py \
--config configs/aggregator/video_aggregator.yml \
configs/ucp/base_ucp.yml \
configs/data/video.yml \
configs/pipeline/video_pipeline/base_pipeline.yml \
configs/extractor/video_extractor/mtcnn_fd.yml \
configs/extractor/video_extractor/iou_tracker.yml \
configs/extractor/video_extractor/hse_emotion.yml \

# python main.py \
# --config configs/pipeline/base_pipeline.yml \
# configs/ucp/base_ucp.yml \
# configs/aggregator/base_aggregator.yml \
# configs/data/video.yml \ 
# configs/pipeline/base_pipeline.yml \


# a video inference pipeline process involves aggregator, ucp, detector, tracker, and emotional representation
