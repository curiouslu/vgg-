# trainer.py
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

def accuracy(logits, y):
    preds = logits.argmax(dim=1)
    return (preds == y).float().mean().item()

class Trainer:
    def __init__(self, n_epochs=10, lr=None, weight_decay=1e-4, device=None,
                 early_stopping_patience=5, grad_clip=None):
        self.n_epochs = n_epochs
        self.weight_decay = weight_decay
        self.grad_clip = grad_clip
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.early_stopping_patience = early_stopping_patience
        self.history = []

    def fit(self, model, train_loader, val_loader, class_weights=None, optimizer=None):
        model.to(self.device)

        opt = optimizer or torch.optim.AdamW(model.parameters(), lr=getattr(model, 'lr', 1e-3), weight_decay=self.weight_decay)
        # Ensure class_weights is a tensor on the correct device
        if hasattr(model, "loss_fn") and isinstance(model.loss_fn, nn.CrossEntropyLoss):
            if class_weights is not None:
                if not torch.is_tensor(class_weights):
                    class_weights = torch.tensor(class_weights, dtype=torch.float32)
                model.loss_fn = nn.CrossEntropyLoss(weight=class_weights.to(self.device))

        writer = SummaryWriter()
        best_val = float("-inf"); wait = 0
        best_state = {k: v.cpu() for k, v in model.state_dict().items()}  # safe fallback

        for epoch in range(1, self.n_epochs + 1):
            # -------- Train --------
            model.train()
            tr_loss = 0.0; tr_acc = 0.0; n = 0
            for xb, yb in train_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                opt.zero_grad(set_to_none=True)
                logits = model(xb)
                loss = model.loss(logits, yb)
                loss.backward()
                if self.grad_clip:
                    nn.utils.clip_grad_norm_(model.parameters(), self.grad_clip)
                opt.step()

                bs = yb.size(0)
                tr_loss += loss.item() * bs
                tr_acc  += accuracy(logits, yb) * bs
                n += bs
            tr_loss /= max(1, n); tr_acc /= max(1, n)

            # -------- Validate --------
            model.eval()
            va_loss = 0.0; va_acc = 0.0; m = 0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(self.device), yb.to(self.device)
                    logits = model(xb)
                    loss = model.loss(logits, yb)
                    bs = yb.size(0)
                    va_loss += loss.item() * bs
                    va_acc  += accuracy(logits, yb) * bs
                    m += bs
            va_loss /= max(1, m); va_acc /= max(1, m)

            # -------- Log (AFTER metrics exist) --------
            writer.add_scalar("Loss/train", tr_loss, epoch)
            writer.add_scalar("Loss/val",   va_loss, epoch)
            writer.add_scalar("Acc/train",  tr_acc,  epoch)
            writer.add_scalar("Acc/val",    va_acc,  epoch)

            print(f"Epoch {epoch:02d} | train_loss {tr_loss:.4f} acc {tr_acc:.4f} | "
                  f"val_loss {va_loss:.4f} acc {va_acc:.4f}")

            self.history.append({
                "epoch": epoch,
                "train_loss": tr_loss, "train_acc": tr_acc,
                "val_loss": va_loss,   "val_acc": va_acc
            })

            # Early stopping on val_acc
            if va_acc > best_val:
                best_val = va_acc
                wait = 0
                best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            else:
                wait += 1
                if wait >= self.early_stopping_patience:
                    print(f"Early stopping at epoch {epoch}")
                    break

        model.load_state_dict(best_state)
        writer.flush(); writer.close()
        return model
