from __future__ import annotations

import argparse
import logging
from pathlib import Path

from torch.utils.data import DataLoader

from Data.dataset import UCFCrimeEventDataset, collate_event


def _configure_logging(level_name: str) -> logging.Logger:
    level = getattr(logging, level_name.upper(), logging.INFO)
    logging.addLevelName(logging.INFO, "info")
    logging.addLevelName(logging.DEBUG, "debug")
    logging.basicConfig(level=level, format="[%(levelname)s] %(message)s")
    return logging.getLogger("ucfcrime.debug")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Debug loader for UCFCrime event feature dataset"
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
        "--classes",
        nargs="*",
        help="Optional subset of class names to load; space separated",
    )
    parser.add_argument(
        "--max-samples-per-class",
        type=int,
        default=None,
        help="Cap the number of samples loaded per class (default: no cap)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Batch size for the debug DataLoader",
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

    data_root = args.data_root / args.variant if args.variant else args.data_root

    dataset = UCFCrimeEventDataset(
        data_root=data_root,
        split=args.split,
        classes=args.classes,
        max_samples_per_class=args.max_samples_per_class,
    )

    num_samples = len(dataset)
    logger.info(
        "dataset variant='%s' split='%s' total_samples=%d",
        args.variant or "<root>",
        args.split,
        num_samples,
    )
    logger.debug("classes=%s", dataset.idx_to_class)

    if num_samples == 0:
        logger.info("dataset is empty; nothing else to inspect")
        return

    first = dataset[0]
    temporal = first["temporal_features"]
    pooled = first["pooled_features"]
    metadata = first["metadata"]

    logger.info(
        "first sample: temporal_shape=%s pooled_shape=%s label=%d",
        tuple(temporal.shape),
        tuple(pooled.shape),
        first["label"],
    )
    logger.debug("first sample metadata=%s", metadata)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_event,
    )

    batch = next(iter(loader))
    temporal_batch = batch["temporal_features"]
    mask = batch["temporal_mask"]
    pooled_batch = batch["pooled_features"]
    labels = batch["labels"]

    logger.info(
        "first batch: temporal_batch_shape=%s mask_shape=%s pooled_batch_shape=%s",
        tuple(temporal_batch.shape),
        tuple(mask.shape),
        tuple(pooled_batch.shape),
    )
    logger.debug("labels=%s", labels.tolist())
    logger.debug("batch metadata examples:")
    for i, meta in enumerate(batch["metadata"], 1):
        logger.debug("  [%d] %s", i, meta)


if __name__ == "__main__":
    main()
