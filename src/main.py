# main.py
import argparse
from data_loader import train_loader, val_loader, class_weights, TARGET_H, TARGET_W, class_to_idx
from model import VGG16FC, VGG16GAP
from trainer import Trainer
from eval_metrics import evaluate
from data_loader import val_loader, class_names
from BasicCNN import TinyCNN
from CNNTrainer import train_tinycnn, plot_training
import torch


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["fc","gap"], default="fc")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--dropout", type=float, default=0.5)
    ap.add_argument("--enforce_49_avg", action="store_true",
                    help="Require 7x7 features for GAP (i.e., input 224x224).")
    return ap.parse_args()

def main():
    args = parse_args()
    in_hw = (TARGET_H, TARGET_W)
    if args.variant == "fc":
        model = VGG16FC(num_classes=len(class_weights), in_hw=in_hw, lr=args.lr, dropout=args.dropout)
    else:
        model = VGG16GAP(num_classes=len(class_weights), in_hw=in_hw, lr=args.lr,
                         enforce_7x7=args.enforce_49_avg)

    trainer = Trainer(n_epochs=args.epochs, weight_decay=args.wd, early_stopping_patience=4)
    model = trainer.fit(model, train_loader, val_loader, class_weights=class_weights)

    cm, ap, mAP = evaluate(model, val_loader)
    print("Confusion matrix:\n", cm)
    print("AP per class:", {cls: round(a, 4) for cls, a in zip(class_names, ap)})
    print("mAP:", round(mAP, 4))

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print("Using device:", device)
    #
    # # ---- Initialise model ----
    # model = TinyCNN(
    #     in_channels=3,  # RGB images
    #     num_classes=len(class_to_idx),
    #     lr=1e-3,
    #     weight_decay=1e-4,
    #     class_weights=class_weights.to(device)
    # )
    #
    # # ---- Train ----
    # model, history = train_tinycnn(
    #     model,
    #     train_loader,
    #     valid_loader=val_loader,
    #     epochs=10,
    #     log_every=20,
    #     device=device,
    #     params={"arch": "TinyCNN", "epochs":10}
    # )
    #
    # # ---- Plot ----
    # plot_training(history)
if __name__ == "__main__":
    main()

