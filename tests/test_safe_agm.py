import torch
from torch.utils.data import DataLoader

from rrnet.agm_data import MEADAGMGroups
from rrnet.safe_agm import SafeAlbedoGenerationModule, SafeAlbedoModel
from rrnet.safe_agm_loss import SafeAGMLoss


def test_safe_agm_is_identity_at_initialization() -> None:
    model = SafeAlbedoModel(shorter_side=64, decoder_channels=16,
                            max_log_gain=1.2).eval()
    image = torch.rand(1, 3, 64, 64)
    with torch.no_grad():
        result = model(image)
    assert torch.allclose(result["albedo"], image, atol=1.0e-6)
    assert torch.allclose(result["gain"], torch.ones_like(result["gain"]),
                          atol=1.0e-6)


def test_safe_agm_preserves_rgb_chromaticity() -> None:
    module = SafeAlbedoGenerationModule(
        feature_channels=(4, 8, 16, 32), decoder_channels=8, max_log_gain=1.0)
    image = torch.tensor([0.20, 0.40, 0.60]).reshape(1, 3, 1, 1).expand(
        1, 3, 32, 32).clone()
    features = [
        torch.rand(1, 4, 16, 16), torch.rand(1, 8, 8, 8),
        torch.rand(1, 16, 4, 4), torch.rand(1, 32, 2, 2),
    ]
    with torch.no_grad():
        module.output.bias.fill_(0.25)
        result = module(image, features)
    before = image / image.sum(dim=1, keepdim=True)
    after = result["albedo"] / result["albedo"].sum(dim=1, keepdim=True)
    assert torch.allclose(after, before, atol=1.0e-5)


def test_safe_agm_prevents_channel_clipping_without_hue_shift() -> None:
    module = SafeAlbedoGenerationModule(
        feature_channels=(4, 8, 16, 32), decoder_channels=8,
        max_log_gain=2.0, output_headroom=0.999)
    image = torch.tensor([0.90, 0.45, 0.225]).reshape(1, 3, 1, 1).expand(
        1, 3, 32, 32).clone()
    features = [
        torch.rand(1, 4, 16, 16), torch.rand(1, 8, 8, 8),
        torch.rand(1, 16, 4, 4), torch.rand(1, 32, 2, 2),
    ]
    with torch.no_grad():
        module.output.bias.fill_(2.0)
        result = module(image, features)
    before = image / image.sum(dim=1, keepdim=True)
    after = result["albedo"] / result["albedo"].sum(dim=1, keepdim=True)
    assert result["albedo"].max() <= 0.999 + 1.0e-6
    assert torch.allclose(after, before, atol=1.0e-5)


def test_safe_agm_mask_keeps_background_identical() -> None:
    model = SafeAlbedoModel(shorter_side=64, decoder_channels=16).eval()
    with torch.no_grad():
        model.agm.output.bias.fill_(0.5)
    image = torch.rand(1, 3, 64, 64) * 0.4
    mask = torch.zeros(1, 1, 64, 64)
    mask[:, :, 16:48, 16:48] = 1.0
    with torch.no_grad():
        result = model(image, mask)
    outside = (1.0 - mask).expand_as(image).bool()
    assert torch.equal(result["albedo"][outside], image[outside])
    assert result["log_gain"].abs().max() <= model.max_log_gain + 1.0e-6


def test_safe_agm_loss_rewards_exact_spatial_inverse() -> None:
    source = torch.full((4, 3, 16, 16), 0.2)
    target = source.clone()
    target[:, :, :, 8:] = 0.4
    mask = torch.ones(4, 1, 16, 16)
    log_gain = ((target[:, :1] + 1.0e-3).log()
                - (source[:, :1] + 1.0e-3).log())
    prediction = {
        "albedo": target.clone(),
        "log_gain": log_gain,
    }
    losses = SafeAGMLoss()(prediction, target, source, mask, mask, group_size=4)
    assert losses["albedo"] < 1.0e-7
    assert losses["gain"] < 1.0e-7
    assert losses["gradient"] < 1.0e-7
    assert losses["beats_global"] > 0.0
    assert torch.isfinite(losses["total"])


def test_identity_anchor_and_loss_on_mead() -> None:
    root = "E:/Lighting Enhancement Project/RRNet/MEAD for RRNet/dataset"
    data = MEADAGMGroups(root, "val", group_size=4,
                         anchor_identity=True, dynamic_epoch=False)
    batch = next(iter(DataLoader(data, batch_size=1)))
    assert batch["identity_flags"].sum() == 1
    source = batch["input"].flatten(0, 1)
    target = batch["target"].flatten(0, 1)
    mask = batch["relight_mask"].flatten(0, 1)
    skin = batch["skin_mask"].flatten(0, 1)
    prediction = {"albedo": source, "log_gain": torch.zeros_like(mask)}
    loss = SafeAGMLoss(lambda_identity=2.0)(
        prediction, target, source, mask, skin, group_size=4,
        identity_flags=batch["identity_flags"].flatten())
    assert loss["identity"] == 0
    assert torch.isfinite(loss["total"])
