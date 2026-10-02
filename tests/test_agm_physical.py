import torch
from PIL import Image

from rrnet.agm_physical_data import MEADPhysicalAGMGroups
from rrnet.agm_physical_stage import PhysicalAGMLoss, oracle_scalar_mae


def test_known_light_loss_is_zero_for_exact_reconstruction():
    source = torch.full((2, 3, 8, 8), 0.4)
    clean = source.clone()
    target = source * 0.8
    mask = torch.ones(2, 1, 8, 8)
    identity = torch.tensor([True, False])
    terms = PhysicalAGMLoss()(
        source=source, clean=clean, target=target,
        albedo=clean, output=target, mask=mask, skin_mask=mask,
        identity_flags=identity)
    assert torch.isclose(terms["total"], torch.tensor(0.0), atol=1e-6)


def test_known_light_loss_backpropagates_through_rgb_agm_output():
    source = torch.full((2, 3, 8, 8), 0.4)
    clean = torch.full_like(source, 0.6)
    albedo = source.clone().requires_grad_()
    mask = torch.ones(2, 1, 8, 8)
    terms = PhysicalAGMLoss()(
        source=source, clean=clean, target=clean,
        albedo=albedo, output=albedo, mask=mask, skin_mask=mask,
        identity_flags=torch.tensor([False, False]))
    terms["total"].backward()
    assert albedo.grad is not None
    assert torch.isfinite(albedo.grad).all()
    assert albedo.grad.abs().sum() > 0


def test_oracle_scalar_has_clean_target_access():
    source = torch.full((1, 3, 8, 8), 0.25)
    target = source * 2.0
    mask = torch.ones(1, 1, 8, 8)
    assert oracle_scalar_mae(source, target, mask).item() < 1e-6


def test_agm_dataset_reads_only_source_person_and_shares_target_theta(tmp_path):
    Image.new("RGB", (8, 8), (90, 100, 110)).save(tmp_path / "clean.png")
    Image.new("L", (8, 8), 255).save(tmp_path / "mask.png")
    (tmp_path / "metadata.csv").write_text(
        "split,degradation_id,clean_frame,relight_mask,skin_mask,person_id,sample_id\n"
        "val,identity,clean.png,mask.png,mask.png,person_a,sample_a\n",
        encoding="utf-8")
    dataset = MEADPhysicalAGMGroups(
        tmp_path, "val", group_size=4, num_lights=1,
        target_lighting_weights={"normal": 1.0})
    sample = dataset[0]
    assert "reference_clean" not in sample
    assert sample["source_clean"].shape == (4, 3, 8, 8)
    assert sample["target_lighting_mode"] == ("normal",) * 4
    assert torch.equal(sample["target_theta"][0], sample["target_theta"][3])
    assert not torch.equal(sample["source_theta"][0], sample["source_theta"][1])
