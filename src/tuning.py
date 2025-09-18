# tuning.py
import os, json, csv, itertools, time
import torch
from torch.utils.data import DataLoader

# import your existing pieces
from data_loader import (
    train_ds, val_ds, class_weights,
    TARGET_H, TARGET_W, class_to_idx
)
from trainer import Trainer
from model import VGG16FC, VGG16GAP

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

ART_DIR = "artifacts"
os.makedirs(ART_DIR, exist_ok=True)
CSV_PATH = os.path.join(ART_DIR, "tuning_results.csv")

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def make_loaders(batch_size):
    # Rebuild loaders so batch_size is tunable
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  pin_memory=False)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, pin_memory=False)
    return train_loader, val_loader

def build_model(cfg, num_classes):
    in_hw = (TARGET_H, TARGET_W)  # comes from your dataloader
    variant = cfg["variant"]
    if variant == "fc":
        return VGG16FC(num_classes=num_classes,
                       in_hw=in_hw,
                       lr=cfg["lr"],
                       dropout=cfg["dropout"],
                       activation=cfg["activation"])
    else:
        enforce_7x7 = (in_hw == (224,224)) and cfg.get("enforce_49", False)
        return VGG16GAP(num_classes=num_classes,
                        in_hw=in_hw,
                        lr=cfg["lr"],
                        activation=cfg["activation"],
                        enforce_7x7=enforce_7x7)

def build_optimizer(cfg, model_params):
    if cfg["optimizer"] == "adamw":
        return torch.optim.AdamW(model_params, lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    elif cfg["optimizer"] == "sgd":
        return torch.optim.SGD(model_params, lr=cfg["lr"], momentum=0.9, weight_decay=cfg["weight_decay"], nesterov=True)
    else:
        raise ValueError(f"Unknown optimizer {cfg['optimizer']}")

def run_trial(cfg, trial_id):
    # Data
    train_loader, val_loader = make_loaders(cfg["batch_size"])

    # Model & opt
    num_classes = len(class_to_idx)
    model = build_model(cfg, num_classes)
    opt = build_optimizer(cfg, model.parameters())

    # Pretty header
    print("\n================ TRIAL {:03d} ================".format(trial_id))
    print(f"Variant        : {cfg['variant'].upper()}")
    print(f"Activation     : {cfg['activation']}")
    print(f"LR             : {cfg['lr']}")
    print(f"Weight Decay   : {cfg['weight_decay']}")
    print(f"Optimizer      : {cfg['optimizer']}")
    print(f"Batch Size     : {cfg['batch_size']}")
    if cfg['variant'] == 'fc': print(f"Dropout        : {cfg['dropout']}")
    if cfg['variant'] == 'gap': print(f"Enforce 49-avg : {cfg.get('enforce_49', False)} (input {TARGET_H}x{TARGET_W})")
    # quick param count
    print(f"Trainable params: {count_params(model):,}")

    # Train
    trainer = Trainer(n_epochs=cfg["epochs"], early_stopping_patience=cfg["patience"])
    start = time.time()
    model = trainer.fit(model, train_loader, val_loader, class_weights=class_weights, optimizer=opt)
    dur = time.time() - start

    # Best epoch
    best = max(trainer.history, key=lambda r: r["val_acc"])
    print(f"Best @ epoch {best['epoch']}: val_acc={best['val_acc']:.4f}, val_loss={best['val_loss']:.4f}")
    print(f"Duration: {dur/60:.1f} min")

    # Return summary
    out = dict(cfg)
    out.update({
        "best_epoch": best["epoch"],
        "best_val_acc": best["val_acc"],
        "best_val_loss": best["val_loss"],
        "params": count_params(model),
        "duration_sec": round(dur, 2),
    })
    return out, model

def write_csv_header_if_needed():
    if not os.path.exists(CSV_PATH):
        with open(CSV_PATH, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "trial","variant","activation","lr","weight_decay","optimizer",
                "batch_size","dropout","epochs","patience","enforce_49",
                "params","best_epoch","best_val_acc","best_val_loss","duration_sec"
            ])

def append_csv_row(trial_id, row):
    with open(CSV_PATH, "a", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            trial_id, row.get("variant"), row.get("activation"), row.get("lr"), row.get("weight_decay"),
            row.get("optimizer"), row.get("batch_size"), row.get("dropout"), row.get("epochs"),
            row.get("patience"), row.get("enforce_49"),
            row.get("params"), row.get("best_epoch"),
            f"{row.get('best_val_acc'):.6f}", f"{row.get('best_val_loss'):.6f}", row.get("duration_sec")
        ])

def main():
    # Define your search space
    grid = {
        "variant": ["gap"],  # stick to GAP for speed
        "activation": ["relu", "gelu"],
        "optimizer": ["adamw"],
        "lr": [3e-4, 1e-4],
        "weight_decay": [0.0, 5e-4],
        "batch_size": [32],  # match data_loader
        "epochs": [10],  # short trials on CPU
        "patience": [3],              # ← add this
        "enforce_49": [True],  # using 224x224 => 7x7 maps
        "dropout": [0.5],  # ignored by GAP but harmless
    }

    # Build the cartesian product
    keys = list(grid.keys())
    space = list(itertools.product(*[grid[k] for k in keys]))

    write_csv_header_if_needed()

    best_row, best_model, best_score = None, None, -1.0
    for trial_id, values in enumerate(space, start=1):
        cfg = dict(zip(keys, values))

        # If variant==gap and enforce_49=True but resize != 224x224, we still run but warn:
        if cfg["variant"] == "gap" and cfg["enforce_49"] and (TARGET_H, TARGET_W) != (224,224):
            print("\n[WARN] enforce_49=True but data_loader is not 224x224; "
                  "the average will be over (TARGET_H/32 * TARGET_W/32) not 49.")

        # Ignore dropout for GAP variant
        if cfg["variant"] == "gap":
            cfg["dropout"] = None

        row, model = run_trial(cfg, trial_id)

        append_csv_row(trial_id, row)

        if row["best_val_acc"] > best_score:
            best_score, best_row, best_model = row["best_val_acc"], row, model
            # save best
            torch.save(best_model.state_dict(), os.path.join(ART_DIR, "best_model.pth"))
            with open(os.path.join(ART_DIR, "best_config.json"), "w") as f:
                json.dump(best_row, f, indent=2)
            print(f"[Saved best so far] val_acc={best_score:.4f}")

    print("\n=========== BEST OVERALL ===========")
    print(json.dumps(best_row, indent=2))
    print("Saved to artifacts/best_config.json and artifacts/best_model.pth")

if __name__ == "__main__":
    main()
