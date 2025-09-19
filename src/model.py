# model.py
import torch
import torch.nn as nn
from typing import Tuple
try:
    from torchvision import models
except Exception:
    models = None  # torchvision may be unavailable in some environments

# ---------------- Activation factory ----------------
def get_activation(name: str):
    name = name.lower().replace("_", "")
    if name == "relu":       return nn.ReLU(inplace=True)
    if name == "leakyrelu":  return nn.LeakyReLU(0.1, inplace=True)
    if name == "gelu":       return nn.GELU()
    if name == "silu" or name == "swish": return nn.SiLU(inplace=True)
    if name == "mish":       return nn.Mish()
    if name == "tanh":       return nn.Tanh()
    raise ValueError(f"Unknown activation: {name}")

# ---------------- From-scratch VGG16 blocks ----------------
def vgg16_features(activation: str = "relu"):
    act = lambda: get_activation(activation)
    return nn.Sequential(
        # Block 1
        nn.Conv2d(3, 64, 3, padding=1), act(),
        nn.Conv2d(64, 64, 3, padding=1), act(),
        nn.MaxPool2d(2, 2),
        # Block 2
        nn.Conv2d(64, 128, 3, padding=1), act(),
        nn.Conv2d(128, 128, 3, padding=1), act(),
        nn.MaxPool2d(2, 2),
        # Block 3
        nn.Conv2d(128, 256, 3, padding=1), act(),
        nn.Conv2d(256, 256, 3, padding=1), act(),
        nn.Conv2d(256, 256, 3, padding=1), act(),
        nn.MaxPool2d(2, 2),
        # Block 4
        nn.Conv2d(256, 512, 3, padding=1), act(),
        nn.Conv2d(512, 512, 3, padding=1), act(),
        nn.Conv2d(512, 512, 3, padding=1), act(),
        nn.MaxPool2d(2, 2),
        # Block 5
        nn.Conv2d(512, 512, 3, padding=1), act(),
        nn.Conv2d(512, 512, 3, padding=1), act(),
        nn.Conv2d(512, 512, 3, padding=1), act(),
        nn.MaxPool2d(2, 2),
    )

class VGG16FC(nn.Module):
    def __init__(self, num_classes: int, in_hw: Tuple[int,int]=(96,128), lr: float=1e-4, dropout: float=0.5, activation: str="relu"):
        super().__init__()
        self.features = vgg16_features(activation)
        H, W = in_hw
        sH, sW = H // 32, W // 32
        fc_in = 512 * sH * sW
        self.classifier = nn.Sequential(
            nn.Linear(fc_in, 4096), get_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(4096, 4096), get_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(4096, num_classes),
        )
        self.lr = lr
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

    def loss(self, outputs, targets): return self.loss_fn(outputs, targets)
    def configure_optimizers(self):   return torch.optim.AdamW(self.parameters(), lr=self.lr)

class VGG16GAP(nn.Module):
    def __init__(self, num_classes: int, in_hw: Tuple[int,int]=(224,224), lr: float=1e-4, activation: str="relu", enforce_7x7: bool=False):
        super().__init__()
        self.features = vgg16_features(activation)
        self.gap = nn.AdaptiveAvgPool2d((1,1))
        self.fc  = nn.Linear(512, num_classes)
        self.lr = lr
        self.loss_fn = nn.CrossEntropyLoss()
        self.enforce_7x7 = enforce_7x7

    def forward(self, x):
        x = self.features(x)           # [B,512,sH,sW]
        if self.enforce_7x7:
            sH, sW = x.shape[-2:]
            if not (sH == 7 and sW == 7):
                raise ValueError("For 49-value average/channel, resize inputs to 224x224.")
        x = self.gap(x).flatten(1)     # [B,512]
        return self.fc(x)

    def loss(self, outputs, targets): return self.loss_fn(outputs, targets)
    def configure_optimizers(self):    return torch.optim.AdamW(self.parameters(), lr=self.lr)

# ---------------- torchvision VGG16 baselines ----------------
class TorchVGG16FC(nn.Module):
    """Wrapper around torchvision.models.vgg16 with a custom classifier head."""
    def __init__(self, num_classes: int, pretrained: bool=False, lr: float=1e-4, dropout: float=0.5):
        super().__init__()
        if models is None:
            raise ImportError("torchvision is not available; cannot use TorchVGG16FC.")
        # We avoid forced download by default (pretrained=False); set True only if weights are available.
        self.backbone = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1 if pretrained else None)
        in_features = self.backbone.classifier[-1].in_features
        # Replace classifier tail
        self.backbone.classifier[-1] = nn.Linear(in_features, num_classes)
        self.lr = lr
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x): return self.backbone(x)
    def loss(self, outputs, targets): return self.loss_fn(outputs, targets)
    def configure_optimizers(self):    return torch.optim.AdamW(self.parameters(), lr=self.lr)

class TorchVGG16GAP(nn.Module):
    """torchvision VGG16 with GAP head (no big FC blocks)."""
    def __init__(self, num_classes: int, pretrained: bool=False, lr: float=1e-4):
        super().__init__()
        if models is None:
            raise ImportError("torchvision is not available; cannot use TorchVGG16GAP.")
        base = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1 if pretrained else None)
        self.features = base.features
        self.gap = nn.AdaptiveAvgPool2d((1,1))
        self.fc  = nn.Linear(512, num_classes)
        self.lr = lr
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        x = self.features(x)
        x = self.gap(x).flatten(1)
        return self.fc(x)

    def loss(self, outputs, targets): return self.loss_fn(outputs, targets)
    def configure_optimizers(self):    return torch.optim.AdamW(self.parameters(), lr=self.lr)
