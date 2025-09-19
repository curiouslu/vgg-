
import argparse, itertools, json, math, os, random, sys, time
from dataclasses import dataclass, asdict
from pathlib import Path
from statistics import mean
from typing import List, Tuple, Dict, Any

import numpy as np
import torch
from sklearn.metrics import average_precision_score, confusion_matrix
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

from data_loader import train_loader, val_loader, input_dim, class_weights, class_to_idx
from trainer import Trainer

# -----------------------------------------------------------------------------
# Config spaces
# -----------------------------------------------------------------------------

def cartesian(space: Dict[str, list]):
    keys, values = zip(*space.items())
    for combo in itertools.product(*values):
        yield dict(zip(keys, combo))

def stage_space(stage: str, base_cfg: dict):
    common = dict(
        variant=base_cfg.get("variant"),
        H=base_cfg.get("H", 224),
        W=base_cfg.get("W", 224),
        grayscale=False,
        flip_p=base_cfg.get("flip_p", 0.0),
    )
    if stage == "act_lr_dropout":
        space = {
            "activation": ["tanh", "mish", "leaky_relu", "silu"],
            "lr": [1e-4],
            "dropout": [0.1, 0.3, 0.5],
        }
        for s in cartesian(space):
            cfg = dict(common); cfg.update(s)
            yield cfg
    elif stage == "augment":
        space = {"flip_p": [0.0, 0.3]}
        for s in cartesian(space):
            cfg = dict(common); cfg.update(s)
            yield cfg
    else:
        raise ValueError(f"Unsupported stage: {stage}")

# -----------------------------------------------------------------------------
# Main run
# -----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=["fc","gap","torch_fc","torch_gap"])
    ap.add_argument("--stage", required=True, choices=["act_lr_dropout","augment"])
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--height", type=int, default=224)
    ap.add_argument("--width", type=int, default=224)
    ap.add_argument("--activation")
    ap.add_argument("--lr", type=float)
    ap.add_argument("--dropout", type=float)
    args = ap.parse_args()

    base_cfg = {
        "variant": args.variant,
        "H": args.height,
        "W": args.width,
        "flip_p": 0.0,
    }

    if args.stage in ("act_lr_dropout","augment"):
        space = list(stage_space(args.stage, base_cfg))
    else:
        raise ValueError(f"Unsupported stage {args.stage}")

    # Normally here you'd train each config; we assume that Trainer handles it
    # Keeping placeholder print for now
    for cfg in space:
        print("Config:", cfg)

if __name__ == "__main__":
    main()
