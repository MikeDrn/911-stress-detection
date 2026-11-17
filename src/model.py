"""ResNet-18 backbones adapted for spectrogram inputs.

Two variants share one checkpoint format and one predictor interface:

* ``resnet18_2d`` (default): torchvision's ResNet-18 with the RGB stem replaced
  by a single-channel 7x7 convolution so it consumes a ``(1, n_mels, frames)``
  Log-Mel "image", and the 1000-way ImageNet head replaced by a 2-way
  ``calm / stressed`` head.
* ``resnet18_1d``: the same 4-stage [2, 2, 2, 2] BasicBlock topology rebuilt
  with ``Conv1d`` so it consumes an MFCC sequence ``(n_mfcc, frames)`` where the
  coefficients are channels and time is the convolution axis.

Class index 1 is "stressed / duress". ``predict_proba`` returns P(class=1),
interpreted as the duress confidence compared against theta in serving.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

ARCHITECTURES = ("resnet18_2d", "resnet18_1d")
INPUT_KEY = {"resnet18_2d": "logmel", "resnet18_1d": "mfcc"}


# --------------------------------------------------------------------------- #
# 2-D variant: torchvision ResNet-18 with spectrogram stem
# --------------------------------------------------------------------------- #
def build_resnet18_2d(num_classes: int = 2, in_channels: int = 1) -> nn.Module:
    from torchvision.models import resnet18

    net = resnet18(weights=None)  # no ImageNet download: spectrograms are not photos
    net.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
    net.fc = nn.Sequential(nn.Dropout(p=0.2), nn.Linear(net.fc.in_features, num_classes))
    return net


# --------------------------------------------------------------------------- #
# 1-D variant: ResNet-18 topology over the time axis
# --------------------------------------------------------------------------- #
class BasicBlock1d(nn.Module):
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.conv2 = nn.Conv1d(planes, planes, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        if stride != 1 or in_planes != planes:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_planes, planes, 1, stride=stride, bias=False),
                nn.BatchNorm1d(planes),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x if self.downsample is None else self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)


class ResNet18_1D(nn.Module):
    def __init__(self, in_channels: int = 20, num_classes: int = 2):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        )
        layers, in_planes = [], 64
        for planes, stride in [(64, 1), (128, 2), (256, 2), (512, 2)]:
            layers.append(
                nn.Sequential(
                    BasicBlock1d(in_planes, planes, stride),
                    BasicBlock1d(planes, planes, 1),
                )
            )
            in_planes = planes
        self.layers = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(nn.Dropout(p=0.2), nn.Linear(512, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layers(self.stem(x))
        return self.fc(torch.flatten(self.pool(x), 1))


# --------------------------------------------------------------------------- #
# Factory, checkpoints and predictor
# --------------------------------------------------------------------------- #
def create_model(architecture: str = "resnet18_2d", num_classes: int = 2, n_mfcc: int = 20):
    if architecture == "resnet18_2d":
        return build_resnet18_2d(num_classes=num_classes)
    if architecture == "resnet18_1d":
        return ResNet18_1D(in_channels=n_mfcc, num_classes=num_classes)
    raise ValueError(f"Unknown architecture '{architecture}'. Choose one of {ARCHITECTURES}.")


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def save_checkpoint(model: nn.Module, path: str | Path, config: dict[str, Any]) -> Path:
    """Persist weights + the minimal config needed to rebuild the network."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "config": dict(config)}, path)
    return path


def load_checkpoint(path: str | Path, map_location: str = "cpu") -> tuple[nn.Module, dict]:
    # weights_only=True: refuse to unpickle arbitrary code from a model file.
    ckpt = torch.load(Path(path), map_location=map_location, weights_only=True)
    cfg = ckpt["config"]
    model = create_model(
        cfg["architecture"], num_classes=cfg.get("num_classes", 2), n_mfcc=cfg.get("n_mfcc", 20)
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, cfg


class TorchStressPredictor:
    """Uniform inference interface shared with ``FakeModel``.

    ``predict_proba(features) -> np.ndarray[n_windows]`` of duress probabilities.
    """

    def __init__(self, model: nn.Module, config: dict[str, Any], version: str = "local"):
        self.model = model.eval()
        self.config = config
        self.architecture = config["architecture"]
        self.input_key = INPUT_KEY[self.architecture]
        self.version = version
        torch.set_num_threads(max(1, torch.get_num_threads()))

    @classmethod
    def from_checkpoint(cls, path: str | Path, version: str = "local") -> TorchStressPredictor:
        model, cfg = load_checkpoint(path)
        return cls(model, cfg, version=version)

    @torch.inference_mode()
    def predict_proba(self, features: dict[str, np.ndarray], batch_size: int = 64) -> np.ndarray:
        x = features[self.input_key]
        out = []
        for i in range(0, len(x), batch_size):
            logits = self.model(torch.from_numpy(np.ascontiguousarray(x[i : i + batch_size])))
            out.append(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
        return np.concatenate(out).astype(np.float32) if out else np.zeros(0, np.float32)
