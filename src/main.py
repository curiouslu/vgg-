# main.py
import argparse
from data_loader import build_dataloaders
from model import VGG16FC, VGG16GAP, TorchVGG16FC, TorchVGG16GAP
from trainer import Trainer
from eval_metrics import evaluate
import torch
import os, csv
import matplotlib.pyplot as plt
import numpy as np
from torchvision.utils import make_grid
import torchvision

# Where to save artifacts (images/CSVs)
ART_DIR = "artifacts"
os.makedirs(ART_DIR, exist_ok=True)

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["fc","gap","torch_fc","torch_gap"], default="fc")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--dropout", type=float, default=0.5)
    ap.add_argument("--activation", type=str, default="relu")
    ap.add_argument("--grayscale", action="store_true", help="Convert images to grayscale (replicated to 3 channels)")
    ap.add_argument("--flip_p", type=float, default=0.0, help="RandomHorizontalFlip probability (0 disables)")
    ap.add_argument("--height", type=int, default=128)
    ap.add_argument("--width", type=int, default=128)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--pretrained", action="store_true", help="Use ImageNet weights for torchvision variants (if available)" )
    ap.add_argument("--enforce_7x7", action="store_true", help="Require 7x7 feature map for GAP (i.e., inputs 224x224)")
    return ap.parse_args()

def main():
    args = parse_args()

    # ---- Data ----
    train_loader, val_loader, meta = build_dataloaders(
        target_hw=(args.height, args.width),
        batch_size=args.batch_size,
        grayscale=args.grayscale,
        flip_p=args.flip_p,
    )
    num_classes = len(meta.class_to_idx)

    # ---- Model ----
    if args.variant == "fc":
        model = VGG16FC(num_classes=num_classes, in_hw=(args.height, args.width),
                        lr=args.lr, dropout=args.dropout, activation=args.activation)
    elif args.variant == "gap":
        model = VGG16GAP(num_classes=num_classes, in_hw=(args.height, args.width),
                         lr=args.lr, activation=args.activation, enforce_7x7=args.enforce_7x7)
    elif args.variant == "torch_fc":
        model = TorchVGG16FC(num_classes=num_classes, pretrained=args.pretrained, lr=args.lr, dropout=args.dropout)
    else:  # torch_gap
        model = TorchVGG16GAP(num_classes=num_classes, pretrained=args.pretrained, lr=args.lr)

    # ---- Train ----
    trainer = Trainer(n_epochs=args.epochs, weight_decay=args.wd, early_stopping_patience=args.patience)
    model = trainer.fit(model, train_loader, val_loader, class_weights=meta.class_weights)

    # ---- Evaluate ----
    cm, ap, mAP = evaluate(model, val_loader)
    print("Confusion matrix:\n", cm)
    print("AP per class:", {cls: round(a, 4) for cls, a in zip(meta.class_names, ap)})
    print("mAP:", round(mAP, 4))

    # --- Save artifacts ---
    # Tag per run for filenames
    tag = args.variant

    # 1) Confusion matrix JPG
    cm_path = os.path.join(ART_DIR, f"confmat_{tag}.jpg")
    save_confmat_jpg(cm, meta.class_names, cm_path)
    print(f"Saved confusion matrix image -> {cm_path}")

    # 2) Per-class AP CSV
    ap_csv = os.path.join(ART_DIR, f"ap_{tag}.csv")
    save_ap_csv(ap, meta.class_names, mAP, ap_csv)
    print(f"Saved per-class AP CSV -> {ap_csv}")

    # 3) Sample predictions grid (+ legend)
    samples_jpg = os.path.join(ART_DIR, f"samples_{tag}.jpg")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    save_sample_predictions(model, val_loader, meta.class_names, samples_jpg, device)
    print(f"Saved sample predictions grid -> {samples_jpg}")

def save_confmat_jpg(cm, labels, out_path):
    fig, ax = plt.subplots(figsize=(6,6))
    im = ax.imshow(cm, interpolation="nearest")
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_xticks(range(len(labels))); ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", rotation_mode="anchor")
    ax.set_yticklabels(labels)
    # write numbers
    vmax = cm.max() if cm.size else 1
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i,j] > 0.6*vmax else "black", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

@torch.no_grad()
def save_sample_predictions(model, loader, labels, out_path, device):
    model.eval()
    xb, yb = next(iter(loader))
    xb = xb.to(device)
    logits = model(xb)
    preds = logits.argmax(dim=1).cpu().numpy()
    y_true = yb.numpy()

    # make a grid of first 16 images with captions
    n = min(16, xb.size(0))
    imgs = xb[:n].cpu()
    grid = make_grid(imgs, nrow=4, normalize=True, value_range=(-1,1))
    # save grid to disk
    torchvision.utils.save_image(grid, out_path)

    # legend text for the first n items
    legend_path = out_path.replace(".jpg", ".txt")
    with open(legend_path, "w") as f:
        for i in range(n):
            f.write(f"{i:02d}: pred={labels[preds[i]]}, true={labels[y_true[i]]}\n")

def save_ap_csv(ap_list, labels, mAP, out_path):
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class","AP"])
        for c, a in zip(labels, ap_list):
            w.writerow([c, float(a)])
        w.writerow(["mAP", float(mAP)])


if __name__ == "__main__":
    main()
