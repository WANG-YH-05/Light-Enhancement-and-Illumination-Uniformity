"""Low-cost face attention derived from an existing full-person mask."""

from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn.functional as F


def _gaussian_blur_mask_tensor(value: torch.Tensor, sigma: float) -> torch.Tensor:
    if sigma <= 0.0:
        return value
    radius = max(1, int(round(3.0 * sigma)))
    coordinates = torch.arange(
        -radius, radius + 1, device=value.device, dtype=value.dtype)
    kernel = torch.exp(-(coordinates * coordinates) / (2.0 * sigma * sigma))
    kernel /= kernel.sum()
    value = F.pad(value, (radius, radius, 0, 0), mode="replicate")
    value = F.conv2d(value, kernel.view(1, 1, 1, -1))
    value = F.pad(value, (0, 0, radius, radius), mode="replicate")
    return F.conv2d(value, kernel.view(1, 1, -1, 1))


def face_attention_from_person_mask_tensor(
        person_mask: torch.Tensor, work_width: int = 512) -> torch.Tensor:
    """CUDA face-attention prior equivalent to the NumPy helper.

    The rectangle and blur are built at a bounded working resolution and then
    upsampled, avoiding a large CPU Gaussian blur on every lighting update.
    """
    if person_mask.ndim != 4 or person_mask.shape[1] != 1:
        raise ValueError("person_mask must have shape [B,1,H,W].")
    batch, _, height, width = person_mask.shape
    small_width = min(work_width, width)
    small_height = max(1, round(height * small_width / width))
    mask = F.interpolate(person_mask.float(), (small_height, small_width),
                         mode="bilinear", align_corners=False)
    attention = torch.zeros_like(mask)
    for index in range(batch):
        support = torch.nonzero(mask[index, 0] > 0.05, as_tuple=False)
        if support.numel() == 0:
            attention[index] = 1.0
            continue
        y0 = int(support[:, 0].min().item())
        y1 = int(support[:, 0].max().item()) + 1
        x0 = int(support[:, 1].min().item())
        x1 = int(support[:, 1].max().item()) + 1
        box_height, box_width = y1 - y0, x1 - x0
        face_x0 = round(x0 + 0.18 * box_width)
        face_x1 = round(x0 + 0.82 * box_width)
        face_y0 = round(y0 + 0.02 * box_height)
        face_y1 = round(y0 + 0.52 * box_height)
        attention[index, 0, face_y0:face_y1, face_x0:face_x1] = 1.0
    sigma = max(1.0, 0.0125 * max(small_height, small_width))
    attention = _gaussian_blur_mask_tensor(attention, sigma) * mask
    maximum = attention.amax(dim=(2, 3), keepdim=True).clamp_min(1.0e-6)
    attention = attention / maximum
    attention = F.interpolate(attention, (height, width), mode="bilinear",
                              align_corners=False)
    return (attention * person_mask.float()).clamp(0.0, 1.0)


def face_attention_from_person_mask(person_mask: np.ndarray) -> np.ndarray:
    """Return a soft upper-central face region without another neural model.

    This mask is used only to encode reference illumination.  It is not used to
    cut out the output person, so its rectangular prior cannot create a visible
    oval or box in the rendered video.
    """
    mask = np.asarray(person_mask, dtype=np.float32)
    if mask.ndim == 3:
        mask = mask[..., 0]
    if mask.ndim != 2:
        raise ValueError("person_mask must be [H,W] or [H,W,1]")
    support = np.argwhere(mask > 0.05)
    if support.size == 0:
        return np.ones(mask.shape + (1,), dtype=np.float32)
    y0, x0 = support.min(axis=0)
    y1, x1 = support.max(axis=0) + 1
    box_height, box_width = y1 - y0, x1 - x0
    face_x0 = int(round(x0 + 0.18 * box_width))
    face_x1 = int(round(x0 + 0.82 * box_width))
    face_y0 = int(round(y0 + 0.02 * box_height))
    face_y1 = int(round(y0 + 0.52 * box_height))
    attention = np.zeros_like(mask)
    attention[face_y0:face_y1, face_x0:face_x1] = 1.0
    sigma = max(1.0, 0.0125 * max(mask.shape))
    attention = cv2.GaussianBlur(attention, (0, 0), sigmaX=sigma, sigmaY=sigma)
    attention *= np.clip(mask, 0.0, 1.0)
    maximum = float(attention.max())
    if maximum > 0.0:
        attention /= maximum
    return attention[..., None].astype(np.float32)


def canonical_face_light_roi(
        rgb: np.ndarray, person_mask: np.ndarray, *,
        context: float = 0.30) -> tuple[np.ndarray, np.ndarray]:
    """Crop a square, face-centred light-estimation input.

    The relighting output remains full resolution.  This helper is only for the
    LPRM light estimator, so a reference portrait and a small face in a 1080p
    meeting frame occupy comparable coordinates before their lighting is
    estimated.  It reuses the person mask already required for compositing;
    no additional detector is introduced.
    """
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("rgb must have shape [H,W,3]")
    if not 0.0 <= context <= 1.0:
        raise ValueError("context must be in [0, 1]")
    mask = np.asarray(person_mask, dtype=np.float32)
    if mask.ndim == 3:
        mask = mask[..., 0]
    if mask.shape != rgb.shape[:2]:
        raise ValueError("person_mask must have the same height and width as rgb")
    support = np.argwhere(mask > 0.05)
    attention = face_attention_from_person_mask(mask)
    if support.size == 0:
        return rgb.copy(), attention

    y0, x0 = support.min(axis=0)
    y1, x1 = support.max(axis=0) + 1
    person_height, person_width = y1 - y0, x1 - x0
    # Match the upper-central prior used by face_attention_from_person_mask,
    # then make it square so scale/aspect-ratio do not leak into LPRM.
    face_x0 = x0 + 0.18 * person_width
    face_x1 = x0 + 0.82 * person_width
    face_y0 = y0 + 0.02 * person_height
    face_y1 = y0 + 0.52 * person_height
    side = max(face_x1 - face_x0, face_y1 - face_y0) * (1.0 + 2.0 * context)
    side = max(2, int(round(side)))
    center_x = 0.5 * (face_x0 + face_x1)
    center_y = 0.5 * (face_y0 + face_y1)
    left = int(round(center_x - 0.5 * side))
    top = int(round(center_y - 0.5 * side))
    right, bottom = left + side, top + side

    pad_left, pad_top = max(0, -left), max(0, -top)
    pad_right = max(0, right - rgb.shape[1])
    pad_bottom = max(0, bottom - rgb.shape[0])
    if pad_left or pad_top or pad_right or pad_bottom:
        rgb = cv2.copyMakeBorder(
            rgb, pad_top, pad_bottom, pad_left, pad_right,
            borderType=cv2.BORDER_REPLICATE)
        attention = cv2.copyMakeBorder(
            attention, pad_top, pad_bottom, pad_left, pad_right,
            borderType=cv2.BORDER_CONSTANT, value=0.0)
        left += pad_left
        right += pad_left
        top += pad_top
        bottom += pad_top
    roi_attention = attention[top:bottom, left:right].copy()
    # OpenCV may squeeze a single-channel image during border operations.
    if roi_attention.ndim == 2:
        roi_attention = roi_attention[..., None]
    return rgb[top:bottom, left:right].copy(), roi_attention
