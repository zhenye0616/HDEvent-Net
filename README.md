# HDEvent-Net

A hyperdimensional computing (HDC) based graph neural network for event-based video understanding. The project implements GrapHD (Graph Hyperdimensional) models that encode graph embeddings using hyperdimensional vectors for knowledge graph completion and event classification tasks.

## Overview

HDEvent-Net combines:
- **Hyperdimensional Computing**: Efficient graph encoding using high-dimensional vector representations
- **Event-CLIP Integration**: Pre-trained vision-language model for event understanding
- **Temporal Modeling**: Processes variable-length event sequences with attention to temporal dynamics

## Dataset Structure

The UCFCrime event dataset should be organized with pre-extracted features:
- Default path: `/mnt/Data_1/UCFCrime_dataset/vitb/`
- Event splits: `event_thr_10/`, `event_thr_25/`
- **14 classes**: Abuse, Arrest, Arson, Assault, Burglary, Explosion, Fighting, Normal, RoadAccidents, Robbery, Shooting, Shoplifting, Stealing, Vandalism
- Features stored as `.npy` files with:
  - Temporal features: `[T, d]` (variable length sequences)
  - Pooled features: `[d]` (mean-pooled over time)

## Setup

### 1. Create Conda Environment

**Create and activate the conda environment:**

```bash
conda create -n eventHD python=3.8
conda activate eventHD
```

**Install PyTorch with CUDA support (required first):**

```bash
pip install torch==2.4.1+cu118 torchvision==0.19.1+cu118 torchaudio==2.4.1+cu118 --index-url https://download.pytorch.org/whl/cu118
```

**Install remaining dependencies:**

```bash
pip install -r requirements.txt
```

**Optional: Install torch-geometric for graph processing:**

```bash
pip install torch-geometric torch-scatter
```

### 2. Download Event-CLIP Checkpoints

The Event-CLIP model checkpoints are available on [HuggingFace](https://huggingface.co/Eavn/event-clip).

**Download the checkpoints:**

```bash
# Create checkpoint directory
mkdir -p ckpt

# Download ViT-B/16 checkpoint (recommended)
wget https://huggingface.co/Eavn/event-clip/resolve/main/vitb.pt -O ckpt/vitb.pt

# Optional: Download ViT-L/14 checkpoint (larger model)
wget https://huggingface.co/Eavn/event-clip/resolve/main/vitl.pt -O ckpt/vitl.pt
```

**Alternative: Using Hugging Face CLI**

```bash
# Install huggingface-hub
pip install huggingface-hub

# Download using CLI
huggingface-cli download Eavn/event-clip vitb.pt --local-dir ckpt
huggingface-cli download Eavn/event-clip vitl.pt --local-dir ckpt
```

The checkpoints should be placed in the `ckpt/` directory:
```
HDEvent-Net/
├── ckpt/
│   ├── vitb.pt    # ViT-B/16 model
│   └── vitl.pt    # ViT-L/14 model (optional)
```

## Usage

### End-to-End Pipeline

The typical workflow processes manifests, segments, attribute scores, and finally produces the KG used for training. Run the following in order when the upstream artefacts change:

1. **Select videos for each split** – `scripts/generate_split.sh`
   - Filters the rich manifest into `Data/UCF_Crime/{train,val,test}.txt` based on class list and augmentation index.
   - Re-run only when you adjust which videos belong to each split.

2. **(Re)segment event streams** – `scripts/build_event_kg.sh`
   - Invokes `Utils/precompute_segments.py` to write canonical segment JSONLs under `Data/segments/<split>/`.
   - Also regenerates structural triples (class/part/precedes). Skip this step if the segmentation config and raw features have not changed.

3. **Score attributes per segment** – `scripts/run_segments.sh`
   - Calls `Utils/attribute_similarity.py` with temporal logits and writes per-segment attribute files to `Data/segment_attributes/<split>/`.
   - Only needs rerunning when you change CLIP scoring parameters or regenerate segments.

4. **Build KG triples (structure + attributes)** – `scripts/build_triples.sh --segment-attrs-root Data/segment_attributes`
   - Reads the segment JSONLs and the corresponding attribute JSONLs to emit `has_attribute` edges alongside structural triples in `Data/UCF_Crime/{train,val,test}.txt`.

5. **Rebuild ID mappings** – `python Utils/build_kg_indices.py ...`
   - Refreshes `entity2id.txt` and `relation2id.txt` so every video/segment/attribute and the `has_attribute` relation receive deterministic IDs.

6. **Train/evaluate** – `python KG.py ...`
   - Loads the refreshed KG files and trains GrapHD/CompGCN variants.

Use the smaller helpers (`build_triples.sh`, `run_attribute_segments.sh`) for incremental updates; the full `build_event_kg.sh` orchestration is convenient when you want to regenerate everything from scratch.

### Dataset Testing

Debug and test the dataset loader:

```bash
python main.py /mnt/Data_1/UCFCrime_dataset --variant vitb --split event_thr_10 --batch-size 10 --log-level debug
```

### Attribute-Event Similarity Analysis

Compute similarity between semantic attributes and event features using Event-CLIP:

**Quick start (uses default settings):**
```bash
bash run_temporal.sh
```

**Using pooled features (mean-pooled over entire video):**
```bash
python Utils/attribute_similarity.py /mnt/Data_1/UCFCrime_dataset \
  --variant vitb \
  --split event_thr_10 \
  --checkpoint ckpt/vitb.pt \
  --classes Abuse \
  --use-logits \
  --top-k 10
```

**Using temporal event segments (analyzes individual segments):**
```bash
python Utils/attribute_similarity.py \
  --checkpoint ckpt/vitb.pt \
  --classes Abuse \
  --use-temporal \
  --use-logits \
  --max-samples 20
```

To export the per-segment attribute scores that feed the KG, supply the segment-related flags:

```bash
python Utils/attribute_similarity.py /mnt/Data_1/UCFCrime_dataset \
  --variant vitb \
  --split event_thr_10 \
  --checkpoint ckpt/vitb.pt \
  --classes Abuse Arrest Arson \
  --use-temporal --use-logits \
  --segment-output-dir Data/segment_attributes \
  --segments-root Data/segments \
  --segments-split train \
  --segment-top-k 5 --segment-aggregate max
```

`scripts/run_attribute_segments.sh` wraps this invocation with project defaults.

**Using cosine similarity instead of CLIP logits:**
```bash
python Utils/attribute_similarity.py \
  --checkpoint ckpt/vitb.pt \
  --classes Abuse \
  --use-temporal \
  --use-similarity
```

**Using ViT-L/14 model:**
```bash
python Utils/attribute_similarity.py \
  --checkpoint ckpt/vitl.pt \
  --model-type "ViT-L/14" \
  --classes Burglary \
  --use-temporal \
  --use-logits
```

### Command-line Options

- `--checkpoint`: Path to Event-CLIP checkpoint (required)
- `--model-type`: CLIP model architecture (`ViT-B/16` or `ViT-L/14`, default: `ViT-B/16`)
- `--classes`: Classes to analyze (e.g., `Abuse Burglary Fighting`)
- `--use-temporal`: Use temporal event segments instead of pooled features
- `--use-logits`: Compute CLIP logits (temperature-scaled similarity, default)
- `--use-similarity`: Use raw cosine similarity without scaling
- `--top-k`: Number of top attributes to display (default: 10)
- `--max-samples`: Maximum samples per class (default: None)

## Architecture

### Core Components

1. **Data Pipeline** (`Data/dataset.py`)
   - `UCFCrimeEventDataset`: Loads pre-extracted event features from `.npy` files
   - `collate_event`: Pads variable-length temporal features to `[B, T_max, d]` with masks
   - Returns: temporal_features, temporal_mask, pooled_features, labels, metadata

2. **GrapHD Model** (`Model/GrapHD.py`)
   - Hyperdimensional encoding layer that transforms entities and relations into HD space
   - Key operations: `mult`, `sub`, `corr` (circular correlation) for binding entity-relation pairs
   - Uses frozen base HDVs with `requires_grad = False`
   - Message passing aggregates: (in_res + out_res + loop_res) / 3

3. **Model Variants** (`Model/models.py`)
   - Base classes: `BaseModel`, `CompGCNBase`, `GrapHDBase`
   - GrapHD models: `GrapHD_DistMult`, `GrapHD_TransE`
   - Legacy CompGCN models: `CompGCN_DistMult`, `CompGCN_TransE`, `CompGCN_ConvE`

4. **Attribute Analysis** (`Utils/attribute_similarity.py`)
   - Loads semantic attributes from `Data/attributes.json` (20 attributes × 10 classes)
   - Encodes attributes using CLIP text encoder with Event-CLIP weights
   - Computes similarity between event embeddings and attribute embeddings
   - Supports both pooled and temporal analysis modes

### Hyperdimensional Encoding Flow

1. Input embeddings (init_embed, init_rel) → HD transformation via learned weights
2. Binding operation (mult/sub/corr) combines entity and relation in HD space
3. Message passing aggregates neighbor information with graph normalization
4. Output: entity embeddings `[num_ent, gcn_dim]` and relation embeddings `[num_rel, gcn_dim]`

## Event-CLIP Integration

The project uses Event-CLIP (from [HuggingFace](https://huggingface.co/Eavn/event-clip)), a momentum contrast (MoCo) framework for event understanding:

- **Event features**: Extracted using encoder_k (key encoder)
- **Text encoding**: CLIP text encoder with Event-CLIP weights
- **Similarity modes**:
  - **CLIP logits** (default): `logit_scale × cosine_similarity` (typical scale ~100)
  - **Cosine similarity**: Raw cosine similarity without temperature scaling

## File Organization

```
HDEvent-Net/
├── main.py                      # Dataset debugging/testing script
├── run.sh                       # Quick start script for attribute analysis
├── Data/
│   ├── dataset.py              # UCFCrime event dataset loader
│   └── attributes.json         # Semantic attributes (10 classes × 20 attributes)
├── Model/
│   ├── GrapHD.py               # Main hyperdimensional GNN layer
│   ├── models.py               # Model variants (DistMult, TransE)
│   ├── compgcn_conv.py         # Standard CompGCN convolution
│   ├── compgcn_conv_basis.py   # Basis decomposition variant
│   └── message_passing.py      # Core message passing framework
├── Utils/
│   └── attribute_similarity.py # Attribute-event similarity analysis
└── ckpt/                       # Event-CLIP checkpoints (download required)
    ├── vitb.pt                 # ViT-B/16 model
    └── vitl.pt                 # ViT-L/14 model (optional)
```
