# main.py
import argparse
import os
import csv
from datetime import datetime

import torch
import numpy as np
import matplotlib.pyplot as plt

from data_loader import build_dataloaders  # returns (train_loader, val_loader, meta)
from model import VGG16FC, VGG16GAP, TorchVGG16FC, TorchVGG16GAP
from trainer import Trainer
from eval_metrics import evaluate  # returns (confusion_matrix, ap_per_class, mAP)

ART_DIR = "artifacts"
os.makedirs(ART_DIR, exist_ok=True)


# ----------------------------- plotting helpers -----------------------------
def _plot_confmat(cm: np.ndarray, class_names, title: str, out_path: str):
    cm = np.array(cm, dtype=np.int64)
    fig, ax = plt.subplots(figsize=(8, 8), dpi=120)
    im = ax.imshow(cm, interpolation="nearest")
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True label",
        xlabel="Predicted label",
        title=title,
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # annotate cells
    thresh = cm.max() / 2.0 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=8,
            )
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _write_ap_csv(ap_list, labels, mAP, out_path):
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class", "AP"])
        for c, a in zip(labels, ap_list):
            w.writerow([c, float(a)])
        w.writerow(["mAP", float(mAP)])


# ----------------------------- build model by variant -----------------------------
def _build_model(variant: str, num_classes: int, in_hw, lr: float, dropout: float, activation: str, pretrained: bool):
    H, W = in_hw
    if variant == "fc":
        return VGG16FC(num_classes=num_classes, in_hw=(H, W), lr=lr, dropout=dropout, activation=activation)
    if variant == "gap":
        return VGG16GAP(num_classes=num_classes, in_hw=(H, W), lr=lr, activation=activation)
    if variant == "torch_fc":
        return TorchVGG16FC(num_classes=num_classes, pretrained=pretrained, lr=lr, dropout=dropout, activation=activation)
    if variant == "torch_gap":
        return TorchVGG16GAP(num_classes=num_classes, pretrained=pretrained, lr=lr, activation=activation)
    raise ValueError(f"Unknown variant: {variant}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["fc", "gap", "torch_fc", "torch_gap"], default="fc")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--wd", type=float, default=1e-4, help="Weight decay for AdamW (used by Trainer)")
    ap.add_argument("--dropout", type=float, default=0.5)
    ap.add_argument("--activation", type=str, default="relu")
    ap.add_argument("--flip_p", type=float, default=0.0, help="RandomHorizontalFlip probability (0 disables)")
    ap.add_argument("--height", type=int, default=224)
    ap.add_argument("--width", type=int, default=224)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--patience", type=int, default=5, help="Early stopping patience (on val_acc)")
    ap.add_argument("--pretrained", action="store_true", help="For torchvision variants: use ImageNet weights if available")
    ap.add_argument("--enforce_7x7", action="store_true", help="If your GAP head expects 7×7 feature map (224×224 inputs)")
    ap.add_argument("--tag", type=str, default="", help="Suffix for saved artifact filenames")
    args = ap.parse_args()

    # -------------- dataloaders --------------
    H, W = args.height, args.width
    grayscale = False  # you asked to remove grayscale search; keep RGB only
    train_loader, val_loader, meta = build_dataloaders(
        size=(H, W),
        batch_size=args.batch_size,
        grayscale=grayscale,
        flip_p=args.flip_p,
        enforce_7x7=args.enforce_7x7,
    )
    class_names = meta.class_names
    num_classes = len(class_names)

    # -------------- model --------------
    model = _build_model(
        variant=args.variant,
        num_classes=num_classes,
        in_hw=(H, W),
        lr=args.lr,
        dropout=args.dropout,
        activation=args.activation,
        pretrained=args.pretrained,
    )

    # -------------- train --------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    trainer = Trainer(
        n_epochs=args.epochs,
        weight_decay=args.wd,
        device=device,
        early_stopping_patience=args.patience,
    )
    model = trainer.fit(model, train_loader, val_loader, class_weights=meta.class_weights)

    # -------------- evaluate on validation --------------
    cm, ap_per_class, mAP = evaluate(model, val_loader, device=device)

    # -------------- save artifacts --------------
    # create a compact slug for filenames
    tag_bits = [args.variant, args.activation, f"lr{args.lr}", f"d{args.dropout}", f"flip{args.flip_p}"]
    if args.tag:
        tag_bits.append(args.tag)
    slug = "_".join(str(b).replace(" ", "") for b in tag_bits)

    cm_path = os.path.join(ART_DIR, f"confmat_{slug}.png")
    _plot_confmat(cm, class_names, title=f"Confusion Matrix — {slug}", out_path=cm_path)
    print(f"Wrote: {cm_path}")

    ap_path = os.path.join(ART_DIR, f"ap_{slug}.csv")
    _write_ap_csv(ap_per_class, class_names, mAP, ap_path)
    print(f"Wrote: {ap_path}")

    # Also write a tiny summary line (handy for grepping)
    summary_path = os.path.join(ART_DIR, f"summary_{slug}.txt")
    with open(summary_path, "w") as f:
        f.write(f"variant={args.variant}\n")
        f.write(f"activation={args.activation}\n")
        f.write(f"lr={args.lr}\n")
        f.write(f"dropout={args.dropout}\n")
        f.write(f"flip_p={args.flip_p}\n")
        f.write(f"epochs={args.epochs} patience={args.patience}\n")
        f.write(f"input={H}x{W} rgb\n")
        f.write(f"mAP={mAP:.4f}\n")
    print(f"Wrote: {summary_path}")


if __name__ == "__main__":
    main()
