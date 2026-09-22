import numpy as np

from tools.evaluate_reference_transfer_grid import psnr, rgb_to_lab, ssim_map


def test_exact_images_have_perfect_psnr_and_ssim() -> None:
    image = np.random.default_rng(7).random((32, 32, 3), dtype=np.float32)
    assert psnr(image, image) >= 119.0
    assert np.allclose(ssim_map(image, image), 1.0, atol=1.0e-5)


def test_lab_uses_expected_black_and_white_lightness() -> None:
    black = rgb_to_lab(np.zeros((1, 1, 3), dtype=np.float32))
    white = rgb_to_lab(np.ones((1, 1, 3), dtype=np.float32))
    assert np.allclose(black, 0.0, atol=1.0e-5)
    assert np.isclose(white[0, 0, 0], 100.0, atol=1.0e-3)
    assert np.allclose(white[0, 0, 1:], 0.0, atol=1.0e-3)
