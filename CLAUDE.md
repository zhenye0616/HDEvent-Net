# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

HDEvent-Net is a hyperdimensional computing (HDC) based graph neural network for event-based video understanding, specifically designed for the UCFCrime dataset. The project implements GrapHD (Graph Hyperdimensional) models that encode graph embeddings using hyperdimensional vectors for knowledge graph completion and event classification tasks.

## Dataset Structure

The UCFCrime event dataset is organized with pre-extracted features:
- Default path: `/mnt/Data_1/UCFCrime_dataset/vitb/`
- Event splits: `event_thr_10/`, `event_thr_25/`
- 14 classes: Abuse, Arrest, Arson, Assault, Burglary, Explosion, Fighting, Normal, RoadAccidents, Robbery, Shooting, Shoplifting, Stealing, Vandalism
- Features stored as `.npy` files with temporal features `[T, d]` and pooled features `[d]`

## Running the Code

Debug/test the dataset loader:
```bash
python main.py /mnt/Data_1/UCFCrime_dataset --variant vitb --split event_thr_10 --batch-size 10 --log-level debug
```

Compute attribute similarity with Event-CLIP text encoder:
```bash
# Run with shell script (uses CLIP logits + temporal segments)
bash run.sh

# Analyze using CLIP logits (default) with pooled features
python Utils/attribute_similarity.py /mnt/Data_1/UCFCrime_dataset --variant vitb --split event_thr_10 \
  --checkpoint /ckpt/vitb.pt --classes Abuse --use-logits --top-k 10

# Analyze using CLIP logits with temporal event segments
python Utils/attribute_similarity.py --checkpoint /ckpt/vitb.pt --classes Abuse \
  --use-temporal --use-logits --max-samples 20

# Use cosine similarity instead of logits
python Utils/attribute_similarity.py --checkpoint /ckpt/vitb.pt --classes Abuse \
  --use-temporal --use-similarity

# Use ViT-L/14 model
python Utils/attribute_similarity.py --checkpoint /ckpt/vitl.pt --model-type "ViT-L/14" \
  --classes Burglary --use-temporal --use-logits
```

## Architecture

### Core Components

1. **Data Pipeline** (`Data/dataset.py`)
   - `UCFCrimeEventDataset`: Loads pre-extracted event features from `.npy` files
   - `collate_event`: Pads variable-length temporal features to `[B, T_max, d]` with masks
   - Returns: temporal_features, temporal_mask, pooled_features, labels, metadata

2. **GrapHD Model** (`Model/GrapHD.py`)
   - Hyperdimensional encoding layer that transforms entities and relations into HD space
   - Key operations: `mult`, `sub`, `corr` (circular correlation) for binding entity-relation pairs
   - Uses frozen base HDVs (w_loop, w_in, w_rel, w_out) with `requires_grad = False`
   - Message passing aggregates: (in_res + out_res + loop_res) / 3

3. **Model Variants** (`Model/models.py`)
   - Base classes: `BaseModel`, `CompGCNBase`, `GrapHDBase`
   - GrapHD models: `GrapHD_DistMult`, `GrapHD_TransE`
   - Quantization models: `GrapHD_Dist_Quant`, `GrapHD_Dist_Quant1`, `GrapHD_Dist_Quant2`, `GrapHD_Dist_EarlyDrop`
   - Legacy CompGCN models: `CompGCN_DistMult`, `CompGCN_TransE`, `CompGCN_ConvE`

4. **Message Passing** (`Model/message_passing.py`)
   - Custom scatter aggregation with configurable reduce operation (sum/mean/max)
   - Currently set to 'mean' reduction (line 25)
   - Note: Can be changed to 'mul' for specific experiments (see commented line 26)

### Key Design Patterns

**Hyperdimensional Encoding Flow:**
1. Input embeddings (init_embed, init_rel) → HD transformation via learned weights
2. Binding operation (mult/sub/corr) combines entity and relation in HD space
3. Message passing aggregates neighbor information with graph normalization
4. Output: entity embeddings `[num_ent, gcn_dim]` and relation embeddings `[num_rel, gcn_dim]`

**Quantization Pipeline:**
- `GrapHD_Dist_Quant`: Dynamic int8 quantization of final MLP layer
- `GrapHD_Dist_Quant2`: Fixed-point quantization of HDVs with configurable bit widths (N_fix, N_frac)
- `GrapHD_Dist_EarlyDrop`: Dimension reduction by selecting high-entropy dimensions

## Important Implementation Details

1. **Model imports require `helper.py`**: All model files import from `helper` module (not included in this repo). This module likely contains:
   - `get_param()`: Parameter initialization utility
   - `ccorr()`: Circular correlation operation
   - PyTorch utilities and common imports

2. **GrapHD vs CompGCN**:
   - GrapHD uses shared weights (w_loop = w_in = w_out = w_rel) for hyperdimensional encoding
   - CompGCN uses separate weights for each direction (w_in, w_out, w_loop, w_rel)
   - GrapHD applies HD transformation before message passing; CompGCN applies it during aggregation

3. **Quantization approaches**:
   - Dynamic quantization: Post-training compression of linear layers
   - Fixed-point: Quantize tensors to fixed bit width using qtorch library
   - Early drop: Reduce dimensionality based on entropy-weighted feature selection

4. **Dataset collation**: Variable-length temporal sequences are padded and masked for batch processing. Always use `collate_event` when creating DataLoaders.

## Dependencies

Core libraries (based on imports):
- PyTorch (torch, torch.nn)
- PyTorch Geometric utilities (torch_scatter)
- NumPy
- qtorch (for quantization experiments)
- scipy (for entropy calculations in dimension dropping)

## Attribute-Based Analysis

The repository includes `Data/attributes.json` with semantic attributes for 10 classes (20 attributes per class). The `attribute_similarity.py` script:

1. Loads text attributes from JSON
2. Loads Event-CLIP model from checkpoint (from [HuggingFace](https://huggingface.co/Eavn/event-clip))
3. Encodes attributes using CLIP text encoder with Event-CLIP weights
4. Computes cosine/dot similarity between event embeddings and attribute embeddings
5. Reports top-k most similar attributes for each class

**Key functions:**
- `load_event_clip_model()`: Loads Event-CLIP checkpoint (vitb.pt or vitl.pt) using encoder_k weights
- `encode_attributes_with_clip()`: Tokenizes and encodes attributes with CLIP text encoder
- `compute_clip_logits()`: Computes CLIP logits (temperature-scaled similarity) mimicking `model(event, text)`
- `compute_similarity()`: Computes similarity matrix (cosine or dot product)
- `analyze_class_similarity()`: Analyzes and reports top attributes per class
  - **Pooled mode** (default): Uses mean-pooled features from entire video `[d]`
  - **Temporal mode** (`--use-temporal`): Uses individual event segments `[T, d]`, averages across segments
  - **Logits mode** (`--use-logits`, default): Computes CLIP logits = `logit_scale × cosine_sim`
  - **Similarity mode** (`--use-similarity`): Uses raw cosine similarity without scaling
  - Provides per-segment statistics when using temporal mode

**Event-CLIP Architecture:**
- Uses momentum contrast (MoCo) framework with encoder_q and encoder_k
- Event features are extracted using encoder_k (key encoder)
- Text encoder shares weights with encoder_k after checkpoint loading
- Logit scale (typically ~100) controls temperature in contrastive learning

## File Organization

```
HDEvent-Net/
├── main.py                   # Dataset debugging/testing script
├── attribute_similarity.py   # Attribute-event similarity analysis
├── Data/
│   ├── dataset.py           # UCFCrime event dataset loader
│   └── attributes.json      # Semantic attributes per class (10 classes × 20 attributes)
├── Model/
│   ├── GrapHD.py            # Main hyperdimensional GNN layer
│   ├── models.py            # Model variants (DistMult, TransE, quantized)
│   ├── compgcn_conv.py      # Standard CompGCN convolution
│   ├── compgcn_conv_basis.py  # Basis decomposition variant
│   └── message_passing.py   # Core message passing framework
└── Utils/                   # (currently empty)
```
