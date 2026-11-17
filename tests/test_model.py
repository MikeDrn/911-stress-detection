"""ResNet-18 adaptation tests (CPU, random weights, no downloads)."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.model import (  # noqa: E402
    TorchStressPredictor,
    count_parameters,
    create_model,
    load_checkpoint,
    save_checkpoint,
)


def test_resnet18_2d_accepts_single_channel_logmel():
    m = create_model("resnet18_2d").eval()
    assert m.conv1.in_channels == 1
    with torch.no_grad():
        out = m(torch.randn(3, 1, 64, 201))
    assert out.shape == (3, 2)
    assert 11_000_000 < count_parameters(m) < 11_300_000  # ResNet-18 scale


def test_resnet18_1d_accepts_mfcc_sequence():
    m = create_model("resnet18_1d", n_mfcc=20).eval()
    with torch.no_grad():
        out = m(torch.randn(4, 20, 201))
    assert out.shape == (4, 2)


def test_unknown_architecture():
    with pytest.raises(ValueError):
        create_model("vgg")


@pytest.mark.parametrize(
    "arch, key, shape",
    [
        ("resnet18_2d", "logmel", (5, 1, 64, 201)),
        ("resnet18_1d", "mfcc", (5, 20, 201)),
    ],
)
def test_checkpoint_roundtrip_and_predictor(tmp_path, arch, key, shape):
    torch.manual_seed(0)
    m = create_model(arch)
    path = save_checkpoint(
        m, tmp_path / "m.pt", {"architecture": arch, "num_classes": 2, "n_mfcc": 20}
    )
    m2, cfg = load_checkpoint(path)
    assert cfg["architecture"] == arch
    x = np.random.default_rng(0).standard_normal(shape).astype(np.float32)
    m.eval()
    with torch.no_grad():
        np.testing.assert_allclose(
            m(torch.from_numpy(x)).numpy(), m2(torch.from_numpy(x)).numpy(), atol=1e-5
        )
    pred = TorchStressPredictor(m2, cfg)
    p = pred.predict_proba({key: x}, batch_size=2)
    assert p.shape == (5,) and ((p >= 0) & (p <= 1)).all()
