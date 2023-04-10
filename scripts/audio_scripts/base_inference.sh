#!/bin/bash

PYTHONPATH='.':$PYTHONPATH \


python main.py \
--config configs/aggregator/audio_aggregator.yml \
configs/data/audio.yml \
configs/pipeline/audio_pipeline/base_pipeline.yml \
configs/extractor/audio_extractor/trill_es.yml \
configs/ucp/base_ucp.yml \

# an audio pipeline inference involves an aggregator, a diarization and emotional representation, and an ucp 
