# tuning.py
import os, json, csv, itertools, time, argparse
import torch

from data_loader import build_dataloaders
from trainer import Trainer
from model import VGG16FC, VGG16GAP, TorchVGG16FC, TorchVGG16GAP

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

ART_DIR = "artifacts"
os.makedirs(ART_DIR, exist_ok=True)

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def build_model(variant, num_classes, cfg):
    if variant == "fc":
        return VGG16FC(num_classes=num_classes, in_hw=(cfg["H"], cfg["W"]),
                       lr=cfg["lr"], dropout=cfg["dropout"], activation=cfg["activation"])
    elif variant == "gap":
        return VGG16GAP(num_classes=num_classes, in_hw=(cfg["H"], cfg["W"]),
                        lr=cfg["lr"], activation=cfg["activation"], enforce_7x7=cfg.get("enforce_7x7", False))
    elif variant == "torch_fc":
        return TorchVGG16FC(num_classes=num_classes, pretrained=cfg.get("pretrained", False),
                            lr=cfg["lr"], dropout=cfg["dropout"])
    elif variant == "torch_gap":
        return TorchVGG16GAP(num_classes=num_classes, pretrained=cfg.get("pretrained", False),
                             lr=cfg["lr"])
    else:
        raise ValueError(f"Unknown variant {variant}")

def run_trial(variant, cfg, trial_id, csv_writer, csv_header_written):
    # ---- Data ----
    train_loader, val_loader, meta = build_dataloaders(
        target_hw=(cfg["H"], cfg["W"]),
        batch_size=cfg["batch_size"],
        grayscale=cfg.get("grayscale", False),
        flip_p=cfg.get("flip_p", 0.0),
    )
    num_classes = len(meta.class_to_idx)

    # ---- Model ----
    model = build_model(variant, num_classes, cfg)

    print("\n================ TRIAL {:03d} ================".format(trial_id))
    print(f"Variant        : {variant.upper()}")
    print(f"Input          : {cfg['H']}x{cfg['W']}  gray={cfg.get('grayscale', False)}  flip_p={cfg.get('flip_p', 0.0)}")
    print(f"Activation     : {cfg['activation']}")
    print(f"LR             : {cfg['lr']}")
    print(f"Dropout        : {cfg.get('dropout')}")
    print(f"Epochs         : {cfg['epochs']}  Patience: {cfg['patience']}")
    print(f"Params         : {count_params(model):,}")

    # ---- Train ----
    trainer = Trainer(n_epochs=cfg["epochs"], early_stopping_patience=cfg["patience"])
    start = time.time()
    model = trainer.fit(model, train_loader, val_loader, class_weights=meta.class_weights)
    dur = time.time() - start

    best = max(trainer.history, key=lambda r: r["val_acc"])
    print(f"Best @ epoch {best['epoch']}: val_acc={best['val_acc']:.4f}, val_loss={best['val_loss']:.4f}")
    print(f"Duration: {dur/60:.1f} min")

    row = {
        "trial": trial_id,
        "variant": variant,
        "H": cfg["H"], "W": cfg["W"],
        "grayscale": cfg.get("grayscale", False),
        "flip_p": cfg.get("flip_p", 0.0),
        "activation": cfg["activation"],
        "lr": cfg["lr"],
        "dropout": cfg.get("dropout"),
        "epochs": cfg["epochs"],
        "patience": cfg["patience"],
        "best_epoch": best["epoch"],
        "best_val_acc": float(best["val_acc"]),
        "best_val_loss": float(best["val_loss"]),
        "duration_sec": round(dur, 2),
    }

    if not csv_header_written[0]:
        csv_writer.writerow(list(row.keys()))
        csv_header_written[0] = True
    csv_writer.writerow(list(row.values()))

    return row, model

def cartesian(dict_of_lists):
    keys = list(dict_of_lists.keys())
    space = list(itertools.product(*[dict_of_lists[k] for k in keys]))
    for vals in space:
        yield dict(zip(keys, vals))

def stage_space(stage, base_cfg):
    """Return a generator of cfg dicts for the requested stage."""
    E = base_cfg["epochs"]; P = base_cfg["patience"]
    common = dict(
        activation=base_cfg.get("activation","relu"),
        lr=base_cfg.get("lr",1e-3),
        dropout=base_cfg.get("dropout",0.5),
        batch_size=base_cfg.get("batch_size",32),
        epochs=E, patience=P,
        grayscale=False, flip_p=0.0,
        H=base_cfg.get("H",128), W=base_cfg.get("W",128),
        enforce_7x7=base_cfg.get("enforce_7x7", False),
        pretrained=base_cfg.get("pretrained", False),
    )

    if stage == "size":
        space = {
            "H": [128, 96],
            "W": [128, 96],
        }
        for s in cartesian(space):
            cfg = dict(common); cfg.update(s)
            yield cfg

    elif stage == "grayscale":
        space = {"grayscale": [False, True]}
        for s in cartesian(space):
            cfg = dict(common); cfg.update(s)
            yield cfg

    elif stage == "act_lr_dropout":
        space = {
            "activation": ["relu", "gelu", "leaky_relu", "silu", "swish", "mish", "tanh"],
            "lr": [1e-3, 1e-4, 3e-3, 1e-2],
            "dropout": [0.1, 0.3, 0.5],
        }
        for s in cartesian(space):
            cfg = dict(common); cfg.update(s)
            yield cfg

    elif stage == "augment":
        space = {"flip_p": [0.3]}
        for s in cartesian(space):
            cfg = dict(common); cfg.update(s)
            yield cfg

    else:
        raise ValueError("Unknown stage")


def parse_args():
    p = argparse.ArgumentParser(description="Multi-stage tuner for VGG variants")
    p.add_argument("--variant", choices=["fc","gap","torch_fc","torch_gap"], required=True,
                   help="Which VGG family to tune.")
    p.add_argument("--stage", choices=["size","grayscale","act_lr_dropout","augment"], required=True,
                   help="Which tuning stage to run.")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--patience", type=int, default=4)
    p.add_argument("--enforce_7x7", action="store_true")
    p.add_argument("--pretrained", action="store_true")
    # Optional overrides to carry best settings between stages
    p.add_argument("--height", type=int, default=128)
    p.add_argument("--width", type=int, default=128)
    p.add_argument("--grayscale", action="store_true")
    p.add_argument("--activation", type=str, default="relu")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--dropout", type=float, default=0.5)
    return p.parse_args()

def main():
    args = parse_args()

    base_cfg = dict(epochs=args.epochs, patience=args.patience,
                    enforce_7x7=args.enforce_7x7, pretrained=args.pretrained,
                    H=args.height, W=args.width, grayscale=args.grayscale,
                    activation=args.activation, lr=args.lr, dropout=args.dropout)

    csv_path = os.path.join(ART_DIR, f"tuning_{args.variant}_{args.stage}.csv")
    best_row, best_model, best_score = None, None, -1.0

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        header_written = [False]

        trial_id = 0
        for cfg in stage_space(args.stage, base_cfg):
            trial_id += 1
            row, model = run_trial(args.variant, cfg, trial_id, writer, header_written)

            if row["best_val_acc"] > best_score:
                best_score, best_row, best_model = row["best_val_acc"], row, model
                torch.save(best_model.state_dict(), os.path.join(ART_DIR, f"best_{args.variant}_{args.stage}.pth"))
                with open(os.path.join(ART_DIR, f"best_{args.variant}_{args.stage}.json"), "w") as jf:
                    json.dump(best_row, jf, indent=2)
                print(f"[Saved best so far] val_acc={best_score:.4f}")

    print("\n=========== BEST for stage ===========")
    print(json.dumps(best_row, indent=2))
    print(f"Saved to {ART_DIR}/best_{args.variant}_{args.stage}.json and .pth")
    print(f"All trials CSV: {csv_path}")

if __name__ == "__main__":
    main()
