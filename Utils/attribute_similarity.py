from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Dict, Optional, Set

import numpy as np
import torch
import torch.nn.functional as F
import clip 

project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from Data.dataset import UCFCrimeEventDataset
from Utils.precompute_segments import load_event_stream

def _configure_logging(level_name: str) -> logging.Logger:
    level = getattr(logging, level_name.upper(), logging.INFO)
    logging.addLevelName(logging.INFO, "info")
    logging.addLevelName(logging.DEBUG, "debug")
    logging.basicConfig(level=level, format="[%(levelname)s] %(message)s")
    return logging.getLogger("attribute_similarity")


def load_attributes(attributes_path: Path) -> Dict[str, List[str]]:
    """Load attributes from JSON file."""
    with open(attributes_path, 'r') as f:
        attributes = json.load(f)
    return attributes


def load_event_clip_model(
    checkpoint_path: Path,
    model_type: str = "ViT-B/32",
    device: str = "cuda"
):
    """
    Load Event-CLIP model from checkpoint.

    Args:
        checkpoint_path: Path to the checkpoint file (vitb.pt or vitl.pt)
        model_type: CLIP model architecture ("ViT-B/32" or "ViT-L/14")
        device: Device to load model on

    Returns:
        Loaded CLIP model
    """

    logger = logging.getLogger("attribute_similarity")
    logger.info(f"Loading CLIP model: {model_type}")

    model, preprocess = clip.load(model_type, device=device)

    if checkpoint_path and checkpoint_path.exists():
        logger.info(f"Loading Event-CLIP checkpoint from: {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, map_location=device)["checkpoint"]

        # Extract encoder_k weights (the key encoder used for event features)
        new_state_dict = {}
        for key in state_dict.keys():
            if 'encoder_k' in key:
                new_state_dict[key.replace('encoder_k.', '')] = state_dict[key]

        model.load_state_dict(new_state_dict)
        logger.info("Checkpoint loaded successfully")
    else:
        logger.warning(f"Checkpoint not found at {checkpoint_path}, using pretrained CLIP")

    return model, preprocess


def encode_attributes_with_clip(
    attributes: Dict[str, List[str]],
    model, preprocess,
    device: str = "cuda"
) -> Dict[str, torch.Tensor]:
    """
    Encode text attributes using CLIP text encoder.

    Args:
        attributes: Dict mapping class names to list of attribute strings
        model: CLIP model instance
        device: Device to run inference on

    Returns:
        Dict mapping class names to encoded attribute tensors [num_attributes, d]
    """

    logger = logging.getLogger("attribute_similarity")
    encoded_attributes = {}

    with torch.no_grad():
        for class_name, attr_list in attributes.items():
            logger.debug(f"Encoding {len(attr_list)} attributes for class '{class_name}'")

            # Tokenize attributes
            text_tokens = clip.tokenize(attr_list).to(device)

            # Encode with text encoder
            text_features = model.encode_text(text_tokens)
            text_features = F.normalize(text_features, p=2, dim=-1)  # L2 normalize

            encoded_attributes[class_name] = text_features.cpu()
            logger.debug(f"  Encoded shape: {text_features.shape}")

    return encoded_attributes


def _sanitize_embeddings(tensor: torch.Tensor) -> torch.Tensor:
    """Replace NaNs/Infs and ensure floating dtype for downstream ops."""
    if not torch.is_floating_point(tensor):
        tensor = tensor.float()
    return torch.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0)


def _safe_normalize(tensor: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """L2 normalize while guarding against zero vectors."""
    tensor = _sanitize_embeddings(tensor)
    eps = torch.finfo(tensor.dtype).eps
    norms = tensor.norm(p=2, dim=dim, keepdim=True).clamp_min(eps)
    return tensor / norms


def _format_sample_id(sample) -> str:
    """
    Prefer human-readable identifiers based on video id and augmentation index.
    Falls back to legacy segment_id if available.
    """
    video_id = getattr(sample, "video_id", None)
    aug_idx = getattr(sample, "augmentation_idx", None)
    if video_id is None:
        return getattr(sample, "segment_id", "<unknown>")
    if aug_idx is None:
        return video_id
    return f"{video_id}:aug{aug_idx}"


def _build_video_entity(
    sample,
    variant: str,
    feature_split: str,
    video_prefix: str = "video",
) -> str:
    """
    Construct the canonical video entity identifier used by the segmenter.
    Format: video:<variant>:<feature_split>:<Class>/<VideoId>__<aug_idx?>
    """
    class_name = getattr(sample, "class_name", None)
    video_id = getattr(sample, "video_id", None)
    aug_idx = getattr(sample, "augmentation_idx", None)

    if class_name is None or video_id is None:
        raise ValueError("Sample missing class_name or video_id; cannot build video entity.")

    aug_suffix = f"__{aug_idx}" if aug_idx is not None else ""

    parts = [video_prefix]
    if variant:
        parts.append(variant)
    if feature_split:
        parts.append(feature_split)
    return ":".join(parts) + f":{class_name}/{video_id}{aug_suffix}"


def _sanitize_video_filename(video_entity: str) -> str:
    """
    Mirror the sanitization used in precompute_segments when writing JSONL files.
    """
    return (
        video_entity.replace("/", "__")
        .replace("\\", "__")
        .replace(":", "_")
        .replace(" ", "_")
    )


def _load_segments_jsonl(path: Path) -> List[Dict]:
    segments: List[Dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            segments.append(json.loads(line))
    return segments


def _aggregate_segment_scores(
    scores: torch.Tensor,
    timestamps_ms: np.ndarray,
    segments: List[Dict],
    attribute_list: List[str],
    top_k: int,
    threshold: Optional[float],
    aggregate: str,
    video_entity: str,
    logger: logging.Logger,
) -> List[Dict]:
    """
    Aggregate attribute scores for each segment using timestamp ranges.
    """
    if scores.device.type != "cpu":
        scores = scores.cpu()

    timestamps = np.asarray(timestamps_ms)
    results: List[Dict] = []

    for segment in segments:
        start_ms = segment.get("start_ms")
        end_ms = segment.get("end_ms")
        if start_ms is None or end_ms is None:
            continue

        mask = (timestamps >= start_ms) & (timestamps <= end_ms)
        indices = np.flatnonzero(mask)
        if indices.size == 0:
            logger.debug(
                "Segment %s in %s matched 0 events (start=%s end=%s)",
                segment.get("seg_name"),
                video_entity,
                start_ms,
                end_ms,
            )
            continue

        idx_tensor = torch.from_numpy(indices).long()
        segment_scores = scores.index_select(0, idx_tensor)

        if aggregate == "mean":
            agg_scores = segment_scores.mean(dim=0)
        elif aggregate == "max":
            agg_scores, _ = segment_scores.max(dim=0)
        else:
            raise ValueError(f"Unknown aggregate method: {aggregate}")

        k = min(top_k, agg_scores.shape[0])
        values, indices_top = torch.topk(agg_scores, k=k)

        top_attributes = []
        for value, attr_idx in zip(values.tolist(), indices_top.tolist()):
            if threshold is not None and value < threshold:
                continue
            top_attributes.append(
                {
                    "name": attribute_list[attr_idx],
                    "score": float(value),
                }
            )

        if not top_attributes:
            continue

        results.append(
            {
                "segment": segment.get("seg_name"),
                "start_ms": start_ms,
                "end_ms": end_ms,
                "n_events": segment.get("n_events"),
                "flags": segment.get("flags", []),
                "top_attributes": top_attributes,
            }
        )

    return results


def compute_similarity(
    event_embeddings: torch.Tensor,
    attribute_embeddings: torch.Tensor,
    metric: str = "cosine"
) -> torch.Tensor:
    """
    Compute similarity between event embeddings and attribute embeddings.

    Args:
        event_embeddings: [N, d] event features
        attribute_embeddings: [M, d] attribute features
        metric: "cosine" or "dot"

    Returns:
        Similarity matrix [N, M]
    """
    # Ensure both tensors have the same dtype and finite values
    if event_embeddings.dtype != attribute_embeddings.dtype:
        attribute_embeddings = attribute_embeddings.to(event_embeddings.dtype)

    event_embeddings = _sanitize_embeddings(event_embeddings)
    attribute_embeddings = _sanitize_embeddings(attribute_embeddings)

    if metric == "cosine":
        # Normalize both embeddings
        event_norm = _safe_normalize(event_embeddings, dim=-1)
        attr_norm = _safe_normalize(attribute_embeddings, dim=-1)
        similarity = torch.mm(event_norm, attr_norm.T)
    elif metric == "dot":
        similarity = torch.mm(event_embeddings, attribute_embeddings.T)
    else:
        raise ValueError(f"Unknown metric: {metric}")

    return torch.nan_to_num(similarity, nan=0.0, posinf=0.0, neginf=0.0)


def compute_clip_logits(
    event_features: torch.Tensor,
    text_features: torch.Tensor,
    logit_scale: float = 100.0
) -> torch.Tensor:
    """
    Compute CLIP logits per event (mimics: logits_per_event, _ = model(event, text)).

    Args:
        event_features: [N, d] event features
        text_features: [M, d] text features
        logit_scale: Temperature scaling factor from model.logit_scale.exp()

    Returns:
        Logits per event [N, M]
    """
    # Ensure both tensors have the same dtype and finite values
    if event_features.dtype != text_features.dtype:
        text_features = text_features.to(event_features.dtype)

    event_features = _safe_normalize(event_features, dim=-1)
    text_features = _safe_normalize(text_features, dim=-1)

    # Compute logits: logit_scale * cosine_similarity
    # This is equivalent to: logits_per_event, _ = model(event, text)
    logits_per_event = logit_scale * torch.mm(event_features, text_features.T)

    return torch.nan_to_num(logits_per_event, nan=0.0, posinf=0.0, neginf=0.0)


def analyze_class_similarity(
    dataset: UCFCrimeEventDataset,
    encoded_attributes: Dict[str, torch.Tensor],
    target_classes: List[str],
    model=None,
    top_k: int = 5,
    use_temporal: bool = False,
    use_logits: bool = False,
    max_samples: int = 10,
    logger: logging.Logger = None,
    *,
    segments_root: Optional[Path] = None,
    segments_split: str = "train",
    segment_output_dir: Optional[Path] = None,
    segment_top_k: int = 5,
    segment_threshold: Optional[float] = None,
    segment_aggregate: str = "mean",
    data_root: Optional[Path] = None,
    feature_split: str = "",
    variant: str = "",
    sample_period_ms: float = 1.0,
):
    """
    Analyze similarity between event embeddings and attributes for target classes.

    Args:
        dataset: UCFCrimeEventDataset instance
        encoded_attributes: Dict of encoded attributes per class
        target_classes: List of class names to analyze
        model: CLIP model (needed for logit_scale if use_logits=True)
        top_k: Number of top attributes to report
        use_temporal: If True, use temporal segments; if False, use pooled features
        use_logits: If True, compute CLIP logits; if False, use cosine similarity
        max_samples: Maximum number of samples to analyze per class
        logger: Logger instance
    """
    if logger is None:
        logger = logging.getLogger("attribute_similarity")

    # Get logit scale from model if using logits
    if use_logits:
        if model and hasattr(model, 'logit_scale'):
            logit_scale = model.logit_scale.exp().item()
            logger.info(f"Using CLIP logits with scale: {logit_scale:.2f}")
        else:
            logit_scale = 100.0
            logger.info(f"Using default logit scale: {logit_scale:.2f}")
    else:
        logit_scale = None

    # Filter dataset for target classes
    class_indices = {cls: dataset.class_to_idx[cls] for cls in target_classes if cls in dataset.class_to_idx}

    if not class_indices:
        logger.error(f"None of the target classes {target_classes} found in dataset")
        return

    logger.info(f"\nAnalyzing for classes: {list(class_indices.keys())}")
    logger.info(f"Using {'temporal segments' if use_temporal else 'pooled features'}")
    logger.info(f"Metric: {'CLIP logits' if use_logits else 'cosine similarity'}")

    enable_segment_output = segment_output_dir is not None and use_temporal
    if enable_segment_output:
        segments_root = Path(segments_root) if segments_root else Path("HDEvent-Net/Data/segments")
        segment_output_dir = Path(segment_output_dir)
        segment_output_dir.mkdir(parents=True, exist_ok=True)
    else:
        segments_root = None
        segment_output_dir = None

    events_data_root = data_root if data_root is not None else dataset.data_root
    segment_cache: Dict[str, List[Dict]] = {}
    event_cache: Dict[str, Dict] = {}
    exported_videos: Set[str] = set()

    for class_name, class_idx in class_indices.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"Class: {class_name}")
        logger.info(f"{'='*60}")

        # Get all samples for this class
        class_samples = [s for s in dataset.samples if s.class_idx == class_idx]

        if not class_samples:
            logger.warning(f"No samples found for class {class_name}")
            continue

        logger.info(f"Number of samples: {len(class_samples)}")

        # Get attributes for this class (handle "Normal Videos" vs "Normal" mismatch)
        attr_key = class_name if class_name in encoded_attributes else f"{class_name} Videos"
        if attr_key not in encoded_attributes:
            logger.warning(f"No attributes found for class {class_name}")
            continue

        attr_embeddings = encoded_attributes[attr_key]
        attribute_list = load_attributes(Path("Data/attributes.json"))[attr_key]

        # Compute similarity for each sample
        all_similarities = []

        for sample in class_samples[:max_samples]:
            # Load event embedding
            data = dataset[dataset.samples.index(sample)]

            if use_temporal:
                # Use temporal features: [T, d]
                temporal_emb = data["temporal_features"]  # [T, d]
                T = temporal_emb.shape[0]

                # Compute logits or similarity for each temporal segment
                if use_logits:
                    scores = compute_clip_logits(temporal_emb, attr_embeddings, logit_scale)  # [T, M]
                else:
                    scores = compute_similarity(temporal_emb, attr_embeddings, metric="cosine")  # [T, M]

                if not torch.isfinite(scores).all():
                    logger.warning(
                        "Non-finite scores detected for sample %s; replacing with zeros",
                        _format_sample_id(sample)
                    )
                scores = torch.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)

                # Average across temporal dimension
                avg_scores = scores.mean(dim=0)  # [M]
                all_similarities.append(avg_scores)

                logger.debug(f"  Sample {_format_sample_id(sample)}: {T} temporal segments")

                if enable_segment_output:
                    try:
                        video_entity = _build_video_entity(sample, variant, feature_split)
                    except ValueError as exc:
                        logger.warning("Skipping segment export: %s", exc)
                        continue

                    if video_entity in exported_videos:
                        continue

                    safe_id = _sanitize_video_filename(video_entity)
                    if segments_root is None:
                        continue
                    segment_file = segments_root / segments_split / f"{safe_id}.jsonl"
                    if not segment_file.exists():
                        logger.debug("Segment file not found for %s (%s)", video_entity, segment_file)
                        continue

                    if video_entity in segment_cache:
                        segments = segment_cache[video_entity]
                    else:
                        segments = _load_segments_jsonl(segment_file)
                        segment_cache[video_entity] = segments

                    if video_entity in event_cache:
                        event_stream = event_cache[video_entity]
                    else:
                        try:
                            event_stream = load_event_stream(
                                video_entity,
                                segments_split,
                                events_data_root,
                                logger,
                                feature_split=feature_split,
                                sample_period_ms=sample_period_ms,
                            )
                        except FileNotFoundError:
                            logger.warning("Could not reload event stream for %s", video_entity)
                            continue
                        event_cache[video_entity] = event_stream

                    scores_cpu = scores.detach().cpu()

                    segment_records = _aggregate_segment_scores(
                        scores=scores_cpu,
                        timestamps_ms=event_stream["timestamps_ms"],
                        segments=segments,
                        attribute_list=attribute_list,
                        top_k=segment_top_k,
                        threshold=segment_threshold,
                        aggregate=segment_aggregate,
                        video_entity=video_entity,
                        logger=logger,
                    )

                    if segment_records:
                        out_dir = segment_output_dir / segments_split
                        out_dir.mkdir(parents=True, exist_ok=True)
                        out_path = out_dir / f"{safe_id}.jsonl"
                        with out_path.open("w", encoding="utf-8") as handle:
                            for record in segment_records:
                                payload = {
                                    "segment": record["segment"],
                                    "video": video_entity,
                                    "start_ms": record["start_ms"],
                                    "end_ms": record["end_ms"],
                                    "n_events": record["n_events"],
                                    "flags": record["flags"],
                                    "top_attributes": record["top_attributes"],
                                }
                                handle.write(json.dumps(payload) + "\n")
                        exported_videos.add(video_entity)
            else:
                # Use pooled features: [d]
                event_emb = data["pooled_features"].unsqueeze(0)  # [1, d]

                # Compute logits or similarity with all attributes
                if use_logits:
                    scores = compute_clip_logits(event_emb, attr_embeddings, logit_scale)  # [1, M]
                else:
                    scores = compute_similarity(event_emb, attr_embeddings, metric="cosine")  # [1, M]

                if not torch.isfinite(scores).all():
                    logger.warning(
                        "Non-finite scores detected for sample %s; replacing with zeros",
                        _format_sample_id(sample)
                    )
                scores = torch.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
                all_similarities.append(scores[0])

        # Average scores across samples
        avg_scores = torch.stack(all_similarities).mean(dim=0)  # [M]
        if not torch.isfinite(avg_scores).all():
            logger.warning("Non-finite average scores encountered; replacing with zeros")
            avg_scores = torch.nan_to_num(avg_scores, nan=0.0, posinf=0.0, neginf=0.0)

        # Get top-k attributes
        top_k_indices = torch.argsort(avg_scores, descending=True)[:top_k]

        metric_name = "logits" if use_logits else "similarity"
        logger.info(f"\nTop {top_k} attributes by {metric_name} (averaged over {len(all_similarities)} samples):")
        for rank, idx in enumerate(top_k_indices, 1):
            attr = attribute_list[idx]
            score = avg_scores[idx].item()
            logger.info(f"  {rank}. {attr:20s} ({metric_name}: {score:.4f})")

        if use_temporal:
            # Also report per-segment statistics
            logger.info(f"\nPer-segment analysis:")
            all_segment_scores = []
            for sample in class_samples[:max_samples]:
                data = dataset[dataset.samples.index(sample)]
                temporal_emb = data["temporal_features"]

                if use_logits:
                    scores = compute_clip_logits(temporal_emb, attr_embeddings, logit_scale)
                else:
                    scores = compute_similarity(temporal_emb, attr_embeddings, metric="cosine")
                if not torch.isfinite(scores).all():
                    logger.warning(
                        "Non-finite per-segment scores detected for sample %s; replacing with zeros",
                        _format_sample_id(sample)
                    )
                scores = torch.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
                all_segment_scores.append(scores)

            # Concatenate all segments from all samples
            all_segments = torch.cat(all_segment_scores, dim=0)  # [total_T, M]
            max_score_per_segment = all_segments.max(dim=1)[0]  # [total_T]

            logger.info(f"  Total segments analyzed: {all_segments.shape[0]}")
            logger.info(f"  Max {metric_name} per segment - mean: {max_score_per_segment.mean():.4f}, std: {max_score_per_segment.std():.4f}")
            logger.info(f"  Max {metric_name} per segment - min: {max_score_per_segment.min():.4f}, max: {max_score_per_segment.max():.4f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute similarity between event embeddings and text attributes"
    )
    parser.add_argument(
        "data_root",
        type=Path,
        nargs="?",
        default=Path("/mnt/Data_1/UCFCrime_dataset"),
        help="Path to the root directory containing the dataset splits",
    )
    parser.add_argument(
        "--variant",
        default="vitb",
        help="Optional subdirectory under data_root (e.g., vitb, vitl)",
    )
    parser.add_argument(
        "--split",
        default="event_thr_10",
        help="Name of the split subdirectory to load (default: event_thr_10)",
    )
    parser.add_argument(
        "--attributes",
        type=Path,
        default=Path("Data/attributes.json"),
        help="Path to attributes JSON file",
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        default=[
            "Abuse",
            "Arrest",
            "Arson",
            "Assault",
            "Burglary",
            "Explosion",
            "Fighting",
            "Normal",
            "RoadAccidents",
            "Robbery",
        ],
        help="Classes to analyze (default: all 10 UCF-Crime categories)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of top attributes to report (default: 5)",
    )
    parser.add_argument(
        "--use-temporal",
        action="store_true",
        help="Use temporal event segments instead of pooled features",
    )
    parser.add_argument(
        "--use-logits",
        action="store_true",
        help="Use CLIP logits (scaled similarity) instead of raw cosine similarity",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=10,
        help="Maximum number of samples to analyze per class (default: 10)",
    )
    parser.add_argument(
        "--segments-root",
        type=Path,
        default=Path("HDEvent-Net/Data/segments"),
        help="Directory containing precomputed segment JSONLs",
    )
    parser.add_argument(
        "--segments-split",
        type=str,
        default="train",
        choices=("train", "val", "test"),
        help="Manifest split aligned with segment JSONLs",
    )
    parser.add_argument(
        "--segment-output-dir",
        type=Path,
        default=None,
        help="If set, write per-segment attribute scores (JSONL) to this directory",
    )
    parser.add_argument(
        "--segment-top-k",
        type=int,
        default=5,
        help="Number of top attributes to keep per segment",
    )
    parser.add_argument(
        "--segment-threshold",
        type=float,
        default=None,
        help="Optional minimum score threshold for segment attributes",
    )
    parser.add_argument(
        "--segment-aggregate",
        choices=("mean", "max"),
        default="mean",
        help="Aggregation method for segment scores",
    )
    parser.add_argument(
        "--sample-period-ms",
        type=float,
        default=1.0,
        help="Sample period (ms) used when reconstructing timestamps",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Path to Event-CLIP checkpoint (vitb.pt or vitl.pt)",
    )
    parser.add_argument(
        "--augmentation-idx",
        type=int,
        default=None,
        help="Only use samples with the specified augmentation index (optional)",
    )
    parser.add_argument(
        "--model-type",
        default="ViT-B/32",
        choices=["ViT-B/32", "ViT-L/14"],
        help="CLIP model architecture (default: ViT-B/32)",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on (default: cuda if available)",
    )
    parser.add_argument(
        "--log-level",
        default="info",
        choices=("debug", "info", "warning", "error", "critical"),
        help="Logging verbosity (default: info)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger = _configure_logging(args.log_level)

    # Load attributes
    logger.info(f"Loading attributes from {args.attributes}")
    attributes = load_attributes(args.attributes)
    logger.info(f"Loaded attributes for {len(attributes)} classes")

    # Load Event-CLIP model
    model, preprocess = load_event_clip_model(
        checkpoint_path=args.checkpoint,
        model_type=args.model_type,
        device=args.device
    )

    # Encode attributes with CLIP text encoder
    logger.info(f"Encoding attributes with Event-CLIP on {args.device}")
    encoded_attributes = encode_attributes_with_clip(
        attributes,
        model=model,
        preprocess=preprocess,
        device=args.device
    )

    # Load dataset
    data_root = args.data_root / args.variant if args.variant else args.data_root
    logger.info(f"Loading dataset from {data_root / args.split}")
    if args.augmentation_idx is not None:
        logger.info(f"Filtering for augmentation index {args.augmentation_idx}")

    dataset = UCFCrimeEventDataset(
        data_root=data_root,
        split=args.split,
        classes=args.classes,
        augmentation_idx=args.augmentation_idx,
    )

    # Analyze similarity
    analyze_class_similarity(
        dataset=dataset,
        encoded_attributes=encoded_attributes,
        target_classes=args.classes,
        model=model,
        top_k=args.top_k,
        use_temporal=args.use_temporal,
        use_logits=args.use_logits,
        max_samples=args.max_samples,
        logger=logger,
        segments_root=args.segments_root,
        segments_split=args.segments_split,
        segment_output_dir=args.segment_output_dir,
        segment_top_k=args.segment_top_k,
        segment_threshold=args.segment_threshold,
        segment_aggregate=args.segment_aggregate,
        data_root=data_root,
        feature_split=args.split,
        variant=args.variant,
        sample_period_ms=args.sample_period_ms,
    )


if __name__ == "__main__":
    main()
