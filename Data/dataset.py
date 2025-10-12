from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Optional
import numpy as np
import torch
from torch.utils.data import Dataset
"""
Process UCFCrime dataset event features using CLIP-based event encoder.

This script loads pre-extracted event features from the UCFCrime dataset
and performs various analyses using the Event-CLIP model from HuggingFace.

Dataset structure:
- /mnt/Data_1/UCFCrime_dataset/vitb/
  ├── event_thr_10/  # Event features with threshold 10
  ├── event_thr_25/  # Event features with threshold 25
  └── rgb/           # RGB features

Each subdirectory contains 14 classes:
- Abuse, Arrest, Arson, Assault, Burglary, Explosion, Fighting
- Normal, RoadAccidents, Robbery, Shooting, Shoplifting, Stealing, Vandalism
"""



CLASS_LIST_DEFAULT = [
    "Abuse","Arrest","Arson","Assault","Burglary","Explosion","Fighting",
    "Normal","RoadAccidents","Robbery","Shooting","Shoplifting","Stealing","Vandalism"
]


@dataclass
class Event_Seg:
    path: Path
    class_idx: int
    class_name: str
    video_id: str
    augmentation_idx: Optional[int]


class UCFCrimeEventDataset(Dataset):
    """
    Loads pre-extracted UCFCrime event features (.npy).
    Returns:
      - temporal_features: FloatTensor [T, d] (T>=1)
      - pooled_features:  FloatTensor [d]
      - label: int (class index)
      - metadata: dict (class_name, file_path, video_id, augmentation_idx)
    """
    def __init__(
        self,
        data_root: Path | str,
        split: str = "event_thr_10",
        classes: Optional[List[str]] = None,
        max_samples_per_class: Optional[int] = None,
        augmentation_idx: Optional[int] = None,
    ):
        self.data_root = Path(data_root)
        self.split = split
        self.split_dir = self.data_root / split
        if not self.split_dir.exists():
            raise ValueError(f"Split directory not found: {self.split_dir}")

        all_classes = sorted([d.name for d in self.split_dir.iterdir() if d.is_dir()])
        if classes is None:
            classes = all_classes
        else:
            requested = list(classes)
            missing = [c for c in requested if c not in all_classes]
            if missing:
                raise ValueError(
                    f"Unknown classes requested: {missing}. Available: {all_classes}"
                )
            classes = requested
        if not classes:
            raise ValueError("No classes provided.")

        # maps
        self.class_to_idx: Dict[str, int] = {c: i for i, c in enumerate(classes)}
        self.idx_to_class: List[str] = classes

        # collect samples
        self.samples: List[Event_Seg] = []
        self.augmentation_idx = augmentation_idx  # Optional filter for augmentation method
        for cname in classes:
            cdir = (self.split_dir / cname)
            files = sorted(cdir.glob("*.npy"))
            added = 0
            for p in files:
                stem = p.stem  # filename w/o .npy
                video_id = stem
                aug_idx: Optional[int] = None
                # Split augmentation marker if present (e.g., Foo__5 → aug_idx=5)
                if "__" in stem:
                    base_candidate, aug_candidate = stem.rsplit("__", 1)
                    if aug_candidate.isdigit():
                        aug_idx = int(aug_candidate)
                        video_id = base_candidate
                if self.augmentation_idx is not None and aug_idx != self.augmentation_idx:
                    continue
                self.samples.append(
                    Event_Seg(
                        path=p,
                        class_idx=self.class_to_idx[cname],
                        class_name=cname,
                        video_id=video_id,
                        augmentation_idx=aug_idx,
                    )
                )
                added += 1
                if max_samples_per_class is not None and added >= max_samples_per_class:
                    break

        print(f"[UCFCrimeEventDataset] {len(self.samples)} samples from {len(classes)} classes in '{self.split}'")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        arr = np.load(s.path).astype(np.float32)  # [T,d] or [d]
        if arr.ndim == 1:
            temporal = arr[None, :]              # [1, d]
        elif arr.ndim == 2:
            temporal = arr                       # [T, d]
        else:
            raise ValueError(f"Unexpected array shape {arr.shape} in {s.path}")

        pooled = temporal.mean(axis=0)           # [d]

        return {
            "temporal_features": torch.from_numpy(temporal),   # [T, d]
            "pooled_features":   torch.from_numpy(pooled),     # [d]
            "label":             s.class_idx,                  # int
            "metadata": {
                "class_name":   s.class_name,
                "file_path":    str(s.path),
                "video_id":     s.video_id,
                "augmentation_idx": s.augmentation_idx,
            }
        }

def collate_event(batch: List[dict]) -> dict:
    """
    Pads variable-length [T,d] to [B,T_max,d] and returns a mask.
    Also stacks pooled features and labels.
    """
    B = len(batch)
    T_max = max(x["temporal_features"].shape[0] for x in batch)
    d = batch[0]["temporal_features"].shape[1]

    feats = torch.zeros(B, T_max, d, dtype=torch.float32)
    mask  = torch.zeros(B, T_max, dtype=torch.bool)
    pooled = torch.zeros(B, d, dtype=torch.float32)
    labels = torch.tensor([x["label"] for x in batch], dtype=torch.long)
    meta   = [x["metadata"] for x in batch]

    for i, x in enumerate(batch):
        t = x["temporal_features"].shape[0]
        feats[i, :t] = x["temporal_features"]
        mask[i, :t]  = True
        pooled[i]    = x["pooled_features"]

    return {
        "temporal_features": feats,   # [B, T_max, d]
        "temporal_mask":     mask,    # [B, T_max] (True where valid)
        "pooled_features":   pooled,  # [B, d]
        "labels":            labels,  # [B]
        "metadata":          meta,    # list of dicts
    }
