#!/bin/bash

# Attribute Similarity Analysis Script
# Run attribute-event similarity analysis using Event-CLIP

# Change to script directory
cd "$(dirname "$0")"

# Configuration
DATA_ROOT="/mnt/Data_1/UCFCrime_dataset"
VARIANT="vitb"
SPLIT="event_thr_10"
CHECKPOINT="/ckpt/vitb.pt"
CLASSES="Abuse"
TOP_K=10
MAX_SAMPLES=20

# Run with temporal segments and cosine similarity
python Utils/attribute_similarity.py $DATA_ROOT \
  --variant $VARIANT \
  --split $SPLIT \
  --checkpoint $CHECKPOINT \
  --classes $CLASSES \
  --use-temporal \
  --max-samples $MAX_SAMPLES \
  --top-k $TOP_K

# Uncomment to run with pooled features instead of temporal
# python Utils/attribute_similarity.py $DATA_ROOT \
#   --variant $VARIANT \
#   --split $SPLIT \
#   --checkpoint $CHECKPOINT \
#   --classes $CLASSES \
#   --max-samples $MAX_SAMPLES \
#   --top-k $TOP_K

# Uncomment to analyze multiple classes
# python Utils/attribute_similarity.py $DATA_ROOT \
#   --variant $VARIANT \
#   --split $SPLIT \
#   --checkpoint $CHECKPOINT \
#   --classes Abuse Arrest Arson Fighting Robbery \
#   --use-temporal \
#   --max-samples $MAX_SAMPLES \
#   --top-k $TOP_K
