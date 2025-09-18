# model.py
import torch
import torch.nn as nn

def get_activation(name: str):
    name = name.lower()
    if name == "relu":      return nn.ReLU(inplace=True)
    if name == "leakyrelu": return nn.LeakyReLU(0.1, inplace=True)
    if name == "gelu":      return nn.GELU()
    raise ValueError(f"Unknown activation: {name}")

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
    def __init__(self, num_classes, in_hw=(96,128), lr=1e-4, dropout=0.5, activation="relu"):
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
    def configure_optimizers(self):   return torch.optim.Adam(self.parameters(), lr=self.lr)

class VGG16GAP(nn.Module):
    def __init__(self, num_classes, in_hw=(224,224), lr=1e-4, activation="relu", enforce_7x7=False):
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
    def configure_optimizers(self):    return torch.optim.Adam(self.parameters(), lr=self.lr)
