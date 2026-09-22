import numpy as np
import torch

from rrnet.data import category_sample_weights
from rrnet.losses import RRNetLoss, luminance
from rrnet.depth import depth_to_normals
from rrnet.lighting import default_parameter_statistics, parameter_dim
from rrnet.model import RRNet
from rrnet.person_mask import (
    MaskEMA,
    composite_person,
    composite_person_tensor,
    refine_person_mask,
    suppress_boundary_gain,
)
from rrnet.reference_data import MEADReferenceTriplets, source_chroma_with_target_luminance
from rrnet.reference_model import ReferenceRRNet
from rrnet.reference_data import _category_probabilities
from rrnet.reference_mask import (
    face_attention_from_person_mask,
    face_attention_from_person_mask_tensor,
)
from rrnet.reference_relative_loss import ReferenceRelativeLoss
from rrnet.reference_relative_model import ReferenceRelativeRRNet
from rrnet.temporal import LightingEMA, ResidualEMA


def test_parameter_shape() -> None:
    mean, std = default_parameter_statistics(9)
    assert mean.shape == std.shape == (parameter_dim(9),)


def test_flat_depth_normals_face_camera() -> None:
    normal = depth_to_normals(torch.ones(2, 1, 16, 16))
    assert torch.allclose(normal[:, 0:2], torch.zeros_like(normal[:, 0:2]))
    assert torch.allclose(normal[:, 2:3], torch.ones_like(normal[:, 2:3]))


def test_ema_equation() -> None:
    ema = LightingEMA(beta=0.8)
    assert torch.equal(ema.update(torch.tensor([[1.0]])), torch.tensor([[1.0]]))
    assert torch.allclose(ema.update(torch.tensor([[0.0]])), torch.tensor([[0.8]]))


def test_residual_ema_smooths_static_and_reacts_to_motion() -> None:
    ema = ResidualEMA(beta=0.8, motion_threshold=0.05)
    guide = torch.zeros(1, 1, 2, 2)
    assert torch.equal(ema.update(torch.ones_like(guide), guide), torch.ones_like(guide))
    assert torch.allclose(ema.update(torch.zeros_like(guide), guide),
                          torch.full_like(guide, 0.8))
    moved_guide = torch.ones_like(guide)
    assert torch.all(ema.update(torch.zeros_like(guide), moved_guide) < 1.0e-6)


def test_forward_smoke() -> None:
    model = RRNet(num_lights=9, shorter_side=64, allow_depth_proxy=True,
                  use_agm=True, agm_channels=16)
    model.eval()
    with torch.no_grad():
        result = model(torch.rand(1, 3, 64, 64))
    assert result["output"].shape == (1, 3, 64, 64)
    assert result["theta"].shape == (1, parameter_dim(9))
    assert torch.isfinite(result["output"]).all()


def test_cached_illumination_matches_renderer_forward() -> None:
    model = RRNet(num_lights=9, shorter_side=64, allow_depth_proxy=True,
                  use_agm=False, agm_channels=16)
    model.eval()
    image = torch.rand(1, 3, 64, 64)
    with torch.no_grad():
        prediction = model(image)
        cached = model.renderer.apply_illumination(image, prediction["illumination"])
    assert torch.allclose(cached, prediction["output"], atol=1e-6, rtol=1e-6)


def test_category_sample_weights_match_requested_probability_mass() -> None:
    rows = [
        {"degradation_id": "identity"},
        {"degradation_id": "dark"},
        {"degradation_id": "dark"},
        {"degradation_id": "over"},
        {"degradation_id": "over"},
    ]
    weights = category_sample_weights(rows, {
        "identity": 0.30,
        "dark": 0.35,
        "over": 0.35,
    })
    assert torch.isclose(weights[0], torch.tensor(0.30, dtype=torch.double))
    assert torch.isclose(weights[1:3].sum(), torch.tensor(0.35, dtype=torch.double))
    assert torch.isclose(weights[3:5].sum(), torch.tensor(0.35, dtype=torch.double))


def test_relight_mask_keeps_background_identical() -> None:
    model = RRNet(num_lights=9, shorter_side=64, allow_depth_proxy=True,
                  use_agm=False, agm_channels=16)
    model.eval()
    image = torch.rand(1, 3, 64, 64)
    mask = torch.zeros(1, 1, 64, 64)
    mask[:, :, 16:48, 16:48] = 1.0
    with torch.no_grad():
        result = model(image, relight_mask=mask)
    outside = (1.0 - mask).expand_as(image).bool()
    assert torch.equal(result["output"][outside], image[outside])
    assert torch.equal(result["loss_output"][outside], image[outside])
    assert "raw_output" in result
    assert "loss_output" in result


def test_mask_aware_loss_is_finite_and_reports_components() -> None:
    model = RRNet(num_lights=9, shorter_side=64, allow_depth_proxy=True,
                  use_agm=False, agm_channels=16)
    model.eval()
    source = torch.rand(1, 3, 64, 64)
    target = torch.rand(1, 3, 64, 64)
    relight = torch.zeros(1, 1, 64, 64)
    relight[:, :, 8:56, 8:56] = 1.0
    skin = torch.zeros_like(relight)
    skin[:, :, 16:48, 16:48] = 1.0
    with torch.no_grad():
        prediction = model(source, relight_mask=relight)
        losses = RRNetLoss()(prediction, target, source, relight, skin)
    expected = {"total", "pixel", "relight", "skin", "highlight",
                "background", "dark", "overexposure_log", "chroma",
                "roi", "regularization"}
    assert set(losses) == expected
    assert all(torch.isfinite(value) for value in losses.values())
    assert losses["background"].item() == 0.0


def test_person_composite_preserves_zero_mask_background() -> None:
    source = np.full((16, 16, 3), 20, dtype=np.uint8)
    enhanced = np.full((16, 16, 3), 220, dtype=np.uint8)
    confidence = np.zeros((8, 8), dtype=np.float32)
    confidence[2:6, 2:6] = 1.0
    mask = refine_person_mask(confidence, (16, 16), close_radius=0,
                              dilate_radius=0, feather=0)
    output = composite_person(source, enhanced, mask)
    outside = mask[..., 0] == 0.0
    inside = mask[..., 0] == 1.0
    assert np.array_equal(output[outside], source[outside])
    assert np.array_equal(output[inside], enhanced[inside])


def test_tensor_person_composite_preserves_background_and_foreground() -> None:
    source = torch.full((1, 3, 8, 8), 0.1)
    enhanced = torch.full_like(source, 0.9)
    mask = torch.zeros((1, 1, 8, 8))
    mask[:, :, 2:6, 2:6] = 1.0
    output = composite_person_tensor(source, enhanced, mask)
    assert torch.equal(output[:, :, :2], source[:, :, :2])
    assert torch.equal(output[:, :, 2:6, 2:6], enhanced[:, :, 2:6, 2:6])


def test_mask_ema() -> None:
    ema = MaskEMA(beta=0.75)
    assert np.array_equal(ema.update(np.ones((2, 2, 1), dtype=np.float32)),
                          np.ones((2, 2, 1), dtype=np.float32))
    expected = np.full((2, 2, 1), 0.75, dtype=np.float32)
    assert np.allclose(ema.update(np.zeros((2, 2, 1), dtype=np.float32)), expected)


def test_boundary_gain_is_zero_at_silhouette_and_one_in_core() -> None:
    mask = np.zeros((32, 32, 1), dtype=np.float32)
    mask[4:28, 4:28] = 1.0
    alpha = suppress_boundary_gain(mask, fade_pixels=5.0)
    assert alpha[4, 16, 0] == 0.0
    assert alpha[16, 16, 0] == 1.0
    assert alpha[0, 0, 0] == 0.0


def test_reference_model_zero_init_matches_canonical_model() -> None:
    model = ReferenceRRNet(num_lights=9, shorter_side=64,
                           allow_depth_proxy=True, use_agm=False,
                           residual_channels=16)
    model.eval()
    source = torch.rand(2, 3, 64, 64)
    reference = torch.rand(2, 3, 64, 64)
    with torch.no_grad():
        canonical = model.safe_base_prediction(source)["output"]
        result = model(source, reference)
    assert torch.allclose(result["output"], canonical, atol=1e-6, rtol=1e-6)
    assert result["luma_residual"].shape == (2, 1, 64, 64)


def test_reference_gain_preserves_chromaticity_without_clipping() -> None:
    model = ReferenceRRNet(num_lights=9, shorter_side=64,
                           allow_depth_proxy=True, use_agm=False,
                           residual_channels=16)
    canonical = torch.tensor([0.8, 0.4, 0.2]).view(1, 3, 1, 1).expand(1, 3, 8, 8)
    base = {
        "output": canonical,
        "depth": torch.ones(1, 1, 8, 8),
        "theta": torch.zeros(1, parameter_dim(9)),
    }
    result = model.apply_luma_residual(
        canonical, base, torch.ones(1, 1, 8, 8), torch.zeros(1, 32))
    output = result["output"]
    assert output.max() <= 1.0
    expected = canonical / canonical.sum(dim=1, keepdim=True)
    actual = output / output.sum(dim=1, keepdim=True)
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)


def test_safe_illumination_has_no_spatial_color_variation() -> None:
    model = ReferenceRRNet(num_lights=9, shorter_side=64,
                           allow_depth_proxy=True, use_agm=False,
                           residual_channels=16, max_global_wb_shift=0.08)
    illumination = torch.rand(2, 3, 12, 10) * 3.0
    safe = model.sanitize_illumination(illumination)
    chroma = safe / luminance(safe).clamp_min(1.0e-4)
    assert torch.all(chroma.amax(dim=(2, 3)) - chroma.amin(dim=(2, 3)) < 1.0e-5)


def test_positive_residual_can_lift_black_pixels() -> None:
    model = ReferenceRRNet(num_lights=9, shorter_side=64,
                           allow_depth_proxy=True, use_agm=False,
                           residual_channels=16)
    canonical = torch.zeros(1, 3, 4, 4)
    base = {"output": canonical, "depth": torch.ones(1, 1, 4, 4),
            "theta": torch.zeros(1, parameter_dim(9))}
    output = model.apply_luma_residual(
        canonical, base, torch.full((1, 1, 4, 4), 0.2), torch.zeros(1, 32))["output"]
    assert torch.allclose(output, torch.full_like(output, 0.2))


def test_reference_face_attention_excludes_lower_body() -> None:
    person = np.zeros((100, 60, 1), dtype=np.float32)
    person[5:95, 5:55] = 1.0
    face = face_attention_from_person_mask(person)
    assert face.shape == person.shape
    assert face[25, 30, 0] > 0.9
    assert face[85, 30, 0] < 0.01


def test_tensor_face_attention_excludes_lower_body() -> None:
    person = torch.zeros((1, 1, 100, 60))
    person[:, :, 5:95, 5:55] = 1.0
    face = face_attention_from_person_mask_tensor(person, work_width=60)
    assert face.shape == person.shape
    assert float(face[0, 0, 25, 30]) > 0.75
    assert float(face[0, 0, 85, 30]) < 0.01


def test_reference_target_keeps_source_chromaticity() -> None:
    clean = np.zeros((8, 8, 3), dtype=np.float32)
    clean[..., 0], clean[..., 1], clean[..., 2] = 0.45, 0.35, 0.25
    lit = np.clip(clean + 0.1, 0.0, 1.0)
    target = source_chroma_with_target_luminance(clean, lit)
    source_chroma = clean / clean.sum(axis=-1, keepdims=True)
    target_chroma = target / target.sum(axis=-1, keepdims=True)
    assert np.allclose(target_chroma, source_chroma, atol=1.0e-6)


def test_reference_dataset_epoch_state_can_advance_or_remain_fixed() -> None:
    dynamic = object.__new__(MEADReferenceTriplets)
    dynamic.dynamic_epoch = True
    dynamic.epoch = 0
    dynamic.set_epoch(3)
    assert dynamic.epoch == 3
    fixed = object.__new__(MEADReferenceTriplets)
    fixed.dynamic_epoch = False
    fixed.epoch = 0
    fixed.set_epoch(3)
    assert fixed.epoch == 0


def test_reference_relative_forward_uses_bounded_ratio_without_residual() -> None:
    model = ReferenceRelativeRRNet(
        num_lights=9, shorter_side=64, allow_depth_proxy=True,
        use_agm=False, min_transfer_gain=0.4, max_transfer_gain=3.0,
    ).eval()
    source = torch.rand(1, 3, 64, 64).clamp_min(0.05)
    reference = torch.rand(1, 3, 64, 64).clamp_min(0.05)
    mask = torch.ones(1, 1, 64, 64)
    with torch.no_grad():
        result = model(source, reference, mask, mask, mask)
    assert result["output"].shape == source.shape
    assert result["source_illumination"].shape == source.shape
    assert result["reference_illumination_on_source"].shape == source.shape
    assert torch.all(result["transfer_gain"] >= 0.4)
    assert torch.all(result["transfer_gain"] <= 3.0)
    assert torch.isfinite(result["output"]).all()
    assert "luma_residual" not in result


def test_reference_relative_loss_reports_illumination_metrics() -> None:
    model = ReferenceRelativeRRNet(
        num_lights=9, shorter_side=64, allow_depth_proxy=True,
        use_agm=False,
    ).eval()
    source_clean = torch.rand(1, 3, 64, 64).mul(0.7).add(0.1)
    reference_clean = torch.rand(1, 3, 64, 64).mul(0.7).add(0.1)
    source = source_clean * 0.6
    reference = reference_clean * 0.9
    target = source_clean * 0.9
    mask = torch.ones(1, 1, 64, 64)
    with torch.no_grad():
        prediction = model(source, reference, mask, mask, mask)
        losses = ReferenceRelativeLoss(
            num_lights=9, lambda_gain=1.0,
            gain_target_min=0.4, gain_target_max=5.0,
        )(
            prediction, target, source, mask, mask,
            source_clean=source_clean,
            reference=reference,
            reference_clean=reference_clean,
            source_light_mask=mask,
            reference_mask=mask,
        )
    expected = {"illum_source", "illum_reference", "illum_target",
                "gain_log", "gain_mean", "gain_face_mean",
                "gain_target_mean", "theta_gap"}
    assert expected.issubset(losses)
    assert all(torch.isfinite(losses[key]) for key in losses)


def test_reference_relative_log_gain_loss_is_zero_for_exact_gain() -> None:
    model = ReferenceRelativeRRNet(
        num_lights=9, shorter_side=64, allow_depth_proxy=True,
        use_agm=False, max_transfer_gain=5.0,
    ).eval()
    source_clean = torch.rand(1, 3, 64, 64).mul(0.6).add(0.2)
    source = source_clean * 0.5
    target = source_clean
    reference = source_clean.clone()
    mask = torch.ones(1, 1, 64, 64)
    with torch.no_grad():
        prediction = model(source, reference, mask, mask, mask)
        prediction["transfer_gain"] = torch.full_like(source, 2.0)
        losses = ReferenceRelativeLoss(
            num_lights=9, lambda_gain=1.0,
            gain_target_min=0.4, gain_target_max=5.0,
        )(
            prediction, target, source, mask, mask,
            source_clean=source_clean,
            reference=reference,
            reference_clean=source_clean,
            source_light_mask=mask,
            reference_mask=mask,
        )
    assert float(losses["gain_log"]) < 1.0e-6
    assert torch.allclose(losses["gain_face_mean"], torch.tensor(2.0))
    assert torch.allclose(losses["gain_target_mean"], torch.tensor(2.0))


def test_symmetric_transfer_loss_penalizes_missing_darken() -> None:
    """Reference transfer must not have a one-sided brighten-only bias."""
    source = torch.full((1, 3, 8, 8), 0.8)
    target = torch.full_like(source, 0.2)
    prediction = {
        "output": source,
        "depth": torch.ones(1, 1, 8, 8),
        "theta": torch.zeros(1, parameter_dim(9)),
    }
    mask = torch.ones(1, 1, 8, 8)
    lift_only = RRNetLoss(lambda_dark=1.0, dark_mode="lift_only")(
        prediction, target, source, mask, mask)
    symmetric = RRNetLoss(lambda_dark=1.0, dark_mode="symmetric")(
        prediction, target, source, mask, mask)
    assert float(lift_only["dark"]) == 0.0
    assert float(symmetric["dark"]) > 0.5


def test_symmetric_transfer_loss_keeps_missing_lift_penalty() -> None:
    source = torch.full((1, 3, 8, 8), 0.2)
    target = torch.full_like(source, 0.8)
    prediction = {
        "output": source,
        "depth": torch.ones(1, 1, 8, 8),
        "theta": torch.zeros(1, parameter_dim(9)),
    }
    mask = torch.ones(1, 1, 8, 8)
    lift_only = RRNetLoss(lambda_dark=1.0, dark_mode="lift_only")(
        prediction, target, source, mask, mask)
    symmetric = RRNetLoss(lambda_dark=1.0, dark_mode="symmetric")(
        prediction, target, source, mask, mask)
    assert float(lift_only["dark"]) > 0.5
    assert torch.allclose(lift_only["dark"], symmetric["dark"])


def test_overexposure_log_loss_penalizes_missing_compression() -> None:
    source = torch.full((1, 3, 8, 8), 0.8)
    target = torch.full_like(source, 0.4)
    mask = torch.ones(1, 1, 8, 8)
    prediction = {
        "output": source,
        "depth": torch.ones(1, 1, 8, 8),
        "theta": torch.zeros(1, parameter_dim(9)),
    }
    losses = RRNetLoss(lambda_overexposure_log=1.0)(
        prediction, target, source, mask, mask)
    assert float(losses["overexposure_log"]) > 0.6


def test_overexposure_log_loss_ignores_brightening_case() -> None:
    source = torch.full((1, 3, 8, 8), 0.2)
    target = torch.full_like(source, 0.7)
    mask = torch.ones(1, 1, 8, 8)
    prediction = {
        "output": source,
        "depth": torch.ones(1, 1, 8, 8),
        "theta": torch.zeros(1, parameter_dim(9)),
    }
    losses = RRNetLoss(lambda_overexposure_log=1.0)(
        prediction, target, source, mask, mask)
    assert float(losses["overexposure_log"]) == 0.0


def test_reference_category_probabilities_normalize_requested_mix() -> None:
    categories = ("identity", "underexposed_cool", "window_backlight")
    probabilities = _category_probabilities(
        categories,
        {"identity": 1.0, "underexposed_cool": 2.0, "window_backlight": 1.0},
        "source_category_weights",
    )
    assert probabilities is not None
    assert torch.allclose(
        torch.from_numpy(probabilities),
        torch.tensor([0.25, 0.50, 0.25], dtype=torch.float64),
    )


def test_reference_relative_dark_log_lift_targets_deep_shadows() -> None:
    model = ReferenceRelativeRRNet(
        num_lights=9, shorter_side=64, allow_depth_proxy=True,
        use_agm=False, max_transfer_gain=5.0,
    ).eval()
    source_clean = torch.full((1, 3, 64, 64), 0.40)
    source = torch.full_like(source_clean, 0.02)
    target = torch.full_like(source_clean, 0.20)
    reference = target.clone()
    mask = torch.ones(1, 1, 64, 64)
    with torch.no_grad():
        prediction = model(source, reference, mask, mask, mask)
        prediction["output"] = source.clone()
        prediction["loss_output"] = source.clone()
        losses = ReferenceRelativeLoss(
            num_lights=9, lambda_dark_log_lift=0.15,
            dark_log_lift_threshold=0.03,
            dark_log_source_max=0.20,
        )(
            prediction, target, source, mask, mask,
            source_clean=source_clean,
            reference=reference,
            reference_clean=source_clean,
            source_light_mask=mask,
            reference_mask=mask,
        )
    assert float(losses["dark_log_lift"]) > 1.0
    assert torch.isfinite(losses["total"])


def test_reference_relative_dark_log_lift_ignores_darken_case() -> None:
    model = ReferenceRelativeRRNet(
        num_lights=9, shorter_side=64, allow_depth_proxy=True,
        use_agm=False, max_transfer_gain=5.0,
    ).eval()
    source_clean = torch.full((1, 3, 64, 64), 0.40)
    source = torch.full_like(source_clean, 0.20)
    target = torch.full_like(source_clean, 0.02)
    reference = target.clone()
    mask = torch.ones(1, 1, 64, 64)
    with torch.no_grad():
        prediction = model(source, reference, mask, mask, mask)
        prediction["output"] = source.clone()
        prediction["loss_output"] = source.clone()
        losses = ReferenceRelativeLoss(
            num_lights=9, lambda_dark_log_lift=0.15,
        )(
            prediction, target, source, mask, mask,
            source_clean=source_clean,
            reference=reference,
            reference_clean=source_clean,
            source_light_mask=mask,
            reference_mask=mask,
        )
    assert float(losses["dark_log_lift"]) == 0.0
