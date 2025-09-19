# tuning.py
import argparse, csv, time
from pathlib import Path

import torch

from data_loader import build_dataloaders            # expects target_hw=(H,W), grayscale, flip_p, batch_size
from model import VGG16FC, VGG16GAP, TorchVGG16FC, TorchVGG16GAP
from trainer import Trainer

ART = Path("artifacts")
ART.mkdir(exist_ok=True)

def stage_banner(msg: str):
    print("\n" + "="*80)
    print(msg)
    print("="*80 + "\n")

def cartesian(space: dict):
    from itertools import product
    keys = list(space.keys())
    for vals in product(*[space[k] for k in keys]):
        yield {k: v for k, v in zip(keys, vals)}

def build_model(variant: str, num_classes: int, in_hw, lr: float, dropout: float, activation: str, pretrained: bool):
    H, W = in_hw
    if variant == "fc":
        return VGG16FC(num_classes=num_classes, in_hw=(H, W), lr=lr, dropout=dropout, activation=activation)
    if variant == "gap":
        return VGG16GAP(num_classes=num_classes, in_hw=(H, W), lr=lr, activation=activation)
    if variant == "torch_fc":
        return TorchVGG16FC(num_classes=num_classes, pretrained=pretrained, lr=lr, dropout=dropout, activation=activation)
    if variant == "torch_gap":
        return TorchVGG16GAP(num_classes=num_classes, pretrained=pretrained, lr=lr)
    raise ValueError(f"Unknown variant: {variant}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["fc","gap","torch_fc","torch_gap"], required=True)
    ap.add_argument("--stage", choices=["act_lr_dropout","augment"], required=True)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--height", type=int, default=224)
    ap.add_argument("--width", type=int, default=224)
    # for augment stage you can pin a base config; otherwise defaults used
    ap.add_argument("--activation", type=str, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--dropout", type=float, default=None)
    ap.add_argument("--pretrained", action="store_true")
    ap.add_argument("--batch_size", type=int, default=32)
    args = ap.parse_args()

    H, W = args.height, args.width
    grayscale = False
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---------- search spaces ----------
    if args.stage == "act_lr_dropout":
        space = {
            "activation": ["tanh", "mish", "leaky_relu", "silu"],
            "lr": [1e-4],
            "dropout": [0.1, 0.3, 0.5],
            "flip_p": [0.0],  # keep augment fixed in this stage
        }
    else:  # augment stage
        space = {
            "activation": [args.activation or "tanh"],
            "lr": [args.lr or 1e-4],
            "dropout": [args.dropout or 0.5],
            "flip_p": [0.0, 0.3],
        }

    out_csv = ART / f"tuning_{args.variant}_{args.stage}.csv"
    write_header = not out_csv.exists()
    fieldnames = ["activation","lr","dropout","flip_p","best_epoch","best_val_acc","best_val_loss","duration_sec"]

    stage_banner(f"[{args.variant.upper()}] {args.stage} @ {H}x{W} RGB")

    with open(out_csv, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            w.writeheader()

        for s in cartesian(space):
            act = str(s["activation"])
            lr  = float(s["lr"])
            do  = float(s["dropout"])
            flip_p = float(s["flip_p"])

            print("Config:", {"variant": args.variant, "H": H, "W": W, "grayscale": False,
                              "flip_p": flip_p, "activation": act, "lr": lr, "dropout": do})

            # data
            train_loader, val_loader, meta = build_dataloaders(
                target_hw=(H, W),
                batch_size=args.batch_size,
                grayscale=grayscale,
                flip_p=flip_p,
            )
            num_classes = len(meta.class_names)

            # model
            model = build_model(
                variant=args.variant,
                num_classes=num_classes,
                in_hw=(H, W),
                lr=lr,
                dropout=do,
                activation=act,
                pretrained=args.pretrained,
            )

            # train (AdamW inside Trainer/model), early stop on val_acc
            trainer = Trainer(
                n_epochs=args.epochs,
                device=device,
                weight_decay=1e-4,
                early_stopping_patience=args.patience,
            )
            t0 = time.time()
            trainer.fit(model, train_loader, val_loader, class_weights=meta.class_weights)
            dur = round(time.time() - t0, 1)

            # best by val_acc
            best_row = max(trainer.history, key=lambda r: r["val_acc"])

            # write row
            w.writerow({
                "activation": act,
                "lr": lr,
                "dropout": do,
                "flip_p": flip_p,
                "best_epoch": int(best_row["epoch"]),
                "best_val_acc": float(best_row["val_acc"]),
                "best_val_loss": float(best_row["val_loss"]),
                "duration_sec": dur,
            })
            f.flush()

    print(f"WROTE -> {out_csv}")

if __name__ == "__main__":
    main()
