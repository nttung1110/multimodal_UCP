#!/bin/bash

PYTHONPATH='.':$PYTHONPATH \


python main.py \
--config configs/aggregator/base_aggregator.yml \
configs/ucp/base_ucp.yml \
configs/data/text.yml \
configs/pipeline/text_pipeline/base_pipeline.yml \
configs/extractor/text_extractor/compm.yml \
configs/ucp/base_ucp.yml \



