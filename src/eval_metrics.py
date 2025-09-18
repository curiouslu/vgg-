# eval_metrics.py
import torch, numpy as np
from sklearn.metrics import confusion_matrix, average_precision_score

@torch.no_grad()
def evaluate(model, loader, device="cuda" if torch.cuda.is_available() else "cpu"):
    model.to(device).eval()
    ys, preds, probs = [], [], []
    for xb, yb in loader:
        logits = model(xb.to(device))
        p = torch.softmax(logits, dim=1).cpu().numpy()
        probs.append(p)
        preds.append(p.argmax(axis=1))
        ys.append(yb.numpy())
    y = np.concatenate(ys)
    y_pred = np.concatenate(preds)
    P = np.concatenate(probs)
    cm = confusion_matrix(y, y_pred)

    # one-vs-rest AP per class, then macro mAP
    y_onehot = np.eye(P.shape[1])[y]
    ap_per_class = [average_precision_score(y_onehot[:, i], P[:, i]) for i in range(P.shape[1])]
    mAP = float(np.mean(ap_per_class))
    return cm, ap_per_class, mAP
