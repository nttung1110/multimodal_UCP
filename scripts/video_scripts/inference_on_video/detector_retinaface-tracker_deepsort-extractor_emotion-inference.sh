#!/bin/bash

PYTHONPATH='.':$PYTHONPATH \


python main.py \
--config configs/aggregator/video_aggregator.yml \
configs/ucp/base_ucp.yml \
configs/data/video.yml \
configs/pipeline/video_pipeline/base_pipeline.yml \
configs/extractor/video_extractor/retinaface_fd.yml \
configs/extractor/video_extractor/ds_tracker.yml \
configs/extractor/video_extractor/hse_emotion.yml \


# a video inference pipeline process involves aggregator, ucp, detector, tracker, and emotional representation
