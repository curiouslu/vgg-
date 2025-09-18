from pathlib import Path
from collections import Counter
import random
from dataclasses import dataclass
from typing import Tuple, List, Dict, Optional

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

# -------------------- Defaults (can be overridden by build_dataloaders) --------------------
ROOT_DIRS      = ["../data/train", "../data/valid"]   # pool labeled data, then split
VAL_FRACTION   = 0.20                                  # 80/20 split
TARGET_H, TARGET_W = 64, 64                            # image size
BATCH_SIZE     = 32
SEED           = 42

# -------------------- Utilities --------------------
def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def _discover_classes(root_dirs: List[str]) -> Tuple[List[str], Dict[str,int]]:
    class_names = sorted({p.name for rd in root_dirs for p in Path(rd).glob("*") if p.is_dir()})
    if not class_names:
        raise RuntimeError("No class folders found under ROOT_DIRS.")
    class_to_idx = {c: i for i, c in enumerate(class_names)}
    return class_names, class_to_idx

def _gather_samples(root_dirs: List[str], class_names: List[str], class_to_idx: Dict[str,int]):
    samples = []
    for rd in root_dirs:
        for c in class_names:
            cdir = Path(rd) / c
            if not cdir.exists():
                continue
            for f in cdir.iterdir():
                if f.is_file():
                    samples.append((str(f), class_to_idx[c]))
    if not samples:
        raise RuntimeError("No images found. Check ROOT_DIRS.")
    return samples

def _stratified_split(samples, val_fraction: float, seed: int):
    by_class = {}
    for i, (_, y) in enumerate(samples):
        by_class.setdefault(y, []).append(i)
    rng = random.Random(seed)
    train_idx, val_idx = [], []
    for y, idxs in by_class.items():
        rng.shuffle(idxs)
        n = len(idxs)
        n_val = max(1, int(round(n * val_fraction)))
        if n > 1 and n_val >= n:
            n_val = n - 1
        val_idx += idxs[:n_val]
        train_idx += idxs[n_val:]
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return [samples[i] for i in train_idx], [samples[i] for i in val_idx]

class ImageDataset(Dataset):
    def __init__(self, samples, transform):
        self.samples = samples
        self.transform = transform
    def __len__(self): return len(self.samples)
    def __getitem__(self, i):
        path, y = self.samples[i]
        with Image.open(path) as img:
            x = img.convert("RGB")
            x = self.transform(x)
        return x, y

def _make_transforms(target_hw: Tuple[int,int], grayscale: bool, flip_p: float):
    resize = transforms.Resize(target_hw, interpolation=transforms.InterpolationMode.BILINEAR)
    if grayscale:
        # Keep 3 channels so models expecting RGB still work (replicate channel)
        to_tensor = transforms.Compose([transforms.Grayscale(num_output_channels=3), transforms.ToTensor()])
    else:
        to_tensor = transforms.ToTensor()
    norm = transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    train_tfms = [resize]
    if flip_p and flip_p > 0:
        train_tfms.append(transforms.RandomHorizontalFlip(p=flip_p))
    train_tfms += [to_tensor, norm]
    val_tfms = [resize, to_tensor, norm]
    return transforms.Compose(train_tfms), transforms.Compose(val_tfms)

@dataclass(frozen=True)
class DataMeta:
    input_dim: int
    class_to_idx: Dict[str,int]
    class_names: List[str]
    class_weights: torch.Tensor
    train_counts: Dict[int,int]

def build_dataloaders(target_hw: Tuple[int,int] = (TARGET_H, TARGET_W),
                      batch_size: int = BATCH_SIZE,
                      grayscale: bool = False,
                      flip_p: float = 0.0,
                      seed: int = SEED,
                      root_dirs: Optional[List[str]] = None,
                      val_fraction: float = VAL_FRACTION):
    """Factory to build train/val DataLoaders with options for size, grayscale, and augmentation."""
    _set_seed(seed)
    root_dirs = root_dirs or ROOT_DIRS
    class_names, class_to_idx = _discover_classes(root_dirs)
    samples = _gather_samples(root_dirs, class_names, class_to_idx)
    train_samples, val_samples = _stratified_split(samples, val_fraction, seed)

    train_tfms, val_tfms = _make_transforms(target_hw, grayscale, flip_p)
    train_ds = ImageDataset(train_samples, train_tfms)
    val_ds   = ImageDataset(val_samples,   val_tfms)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  pin_memory=False)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, pin_memory=False)

    # Class weights (inverse-frequency)
    train_counts = Counter([y for _, y in train_samples])
    K = len(class_to_idx)
    N = sum(train_counts.values())
    class_weights = torch.tensor([N / (K * max(1, train_counts.get(i, 0))) for i in range(K)], dtype=torch.float32)
    meta = DataMeta(
        input_dim=3 * target_hw[0] * target_hw[1],
        class_to_idx=class_to_idx,
        class_names=class_names,
        class_weights=class_weights,
        train_counts=dict(train_counts),
    )
    return train_loader, val_loader, meta

# -------------------- Backward-compatible module-level objects --------------------
# Build defaults so existing imports keep working.
train_loader, val_loader, _meta = build_dataloaders()
class_to_idx = _meta.class_to_idx
class_names  = _meta.class_names
class_weights = _meta.class_weights
input_dim     = _meta.input_dim

# Helpful prints
print(f"classes: {class_to_idx}")
print(f"train size: {len(train_loader.dataset)}   val size: {len(val_loader.dataset)}")
print("train counts:", _meta.train_counts)
print("input_dim   :", input_dim)
print("class_weights:", class_weights.tolist())
