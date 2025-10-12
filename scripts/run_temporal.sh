#!/bin/bash

# Attribute Similarity Analysis with CLIP Logits
# Computes logits_per_event (mimicking model(event, text))

# Change to project root
cd "$(dirname "$0")/.."

# Configuration
DATA_ROOT="/mnt/Data_1/UCFCrime_dataset"
VARIANT="vitb"
SPLIT="event_thr_10"
CHECKPOINT="/home/biaslab/Zhen/HDEvent-Net/ckpt/vitb.pt"
CLASSES="Abuse Arrest Arson "
#CLASSES="Abuse Arrest Arson Assault Burglary Explosion Fighting RoadAccidents Robbery"
TOP_K=20
AUG_IDX=5
MAX_SAMPLES=50

# Run with CLIP logits and temporal segments
python Utils/attribute_similarity.py $DATA_ROOT \
  --variant $VARIANT \
  --split $SPLIT \
  --checkpoint $CHECKPOINT \
  --classes $CLASSES \
  --use-temporal \
  --use-logits \
  --max-samples $MAX_SAMPLES \
  --augmentation-idx $AUG_IDX \
  --top-k $TOP_K \
  --log-level info

# Uncomment below to run with pooled features instead
# python attribute_similarity.py $DATA_ROOT \
#   --variant $VARIANT \
#   --split $SPLIT \
#   --checkpoint $CHECKPOINT \
#   --classes $CLASSES \
#   --max-samples $MAX_SAMPLES \
#   --top-k $TOP_K

# Uncomment to analyze multiple classes
# python attribute_similarity.py $DATA_ROOT \
#   --variant $VARIANT \
#   --split $SPLIT \
#   --checkpoint $CHECKPOINT \
#   --classes Abuse Arrest Arson Fighting Robbery \
#   --use-temporal \
#   --max-samples $MAX_SAMPLES \
#   --top-k $TOP_K
