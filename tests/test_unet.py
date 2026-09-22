import torch

from src.models.unet import UNet


def test_unet_preserves_spatial_shape():
    model = UNet(in_channels=3, out_channels=1, channels=(8, 16, 32, 64))
    x = torch.randn(2, 3, 64, 64)
    y = model(x)
    assert y.shape == (2, 1, 64, 64)
