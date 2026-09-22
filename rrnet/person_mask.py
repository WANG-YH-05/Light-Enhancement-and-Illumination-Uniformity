"""Lightweight temporally-smoothed person masks for video relighting."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F


def _smoothstep(value: np.ndarray) -> np.ndarray:
    value = np.clip(value, 0.0, 1.0)
    return value * value * (3.0 - 2.0 * value)


def _guided_filter(confidence: np.ndarray, guidance_rgb: np.ndarray,
                   radius: int, epsilon: float) -> np.ndarray:
    """Snap a low-resolution confidence matte to edges in the source frame."""
    if radius <= 0:
        return confidence
    if guidance_rgb.ndim != 3 or guidance_rgb.shape[2] != 3:
        raise ValueError("guidance_rgb must have shape [H, W, 3].")
    guidance = cv2.cvtColor(guidance_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    if guidance.max() > 1.0:
        guidance /= 255.0
    if guidance.shape != confidence.shape:
        guidance = cv2.resize(
            guidance, (confidence.shape[1], confidence.shape[0]),
            interpolation=cv2.INTER_LINEAR)
    size = (2 * radius + 1, 2 * radius + 1)
    mean_i = cv2.boxFilter(guidance, -1, size, borderType=cv2.BORDER_REFLECT)
    mean_p = cv2.boxFilter(confidence, -1, size, borderType=cv2.BORDER_REFLECT)
    corr_i = cv2.boxFilter(guidance * guidance, -1, size,
                           borderType=cv2.BORDER_REFLECT)
    corr_ip = cv2.boxFilter(guidance * confidence, -1, size,
                            borderType=cv2.BORDER_REFLECT)
    variance_i = corr_i - mean_i * mean_i
    covariance_ip = corr_ip - mean_i * mean_p
    a = covariance_ip / (variance_i + epsilon)
    b = mean_p - a * mean_i
    mean_a = cv2.boxFilter(a, -1, size, borderType=cv2.BORDER_REFLECT)
    mean_b = cv2.boxFilter(b, -1, size, borderType=cv2.BORDER_REFLECT)
    return np.clip(mean_a * guidance + mean_b, 0.0, 1.0)


def refine_person_mask(confidence: np.ndarray, output_shape: tuple[int, int], *,
                       guidance_rgb: np.ndarray | None = None,
                       threshold_low: float = 0.30,
                       threshold_high: float = 0.70,
                       close_radius: int = 2,
                       dilate_radius: int = 0,
                       feather: float = 1.5,
                       guided_radius: int = 5,
                       guided_epsilon: float = 1.0e-3,
                       edge_power: float = 1.5) -> np.ndarray:
    """Convert a person-confidence map into a clean soft alpha mask [H,W,1]."""
    if not 0.0 <= threshold_low < threshold_high <= 1.0:
        raise ValueError("Mask thresholds must satisfy 0 <= low < high <= 1.")
    if close_radius < 0 or dilate_radius < 0 or feather < 0 or guided_radius < 0:
        raise ValueError("Mask morphology and feather values must be non-negative.")
    if guided_epsilon <= 0.0 or edge_power <= 0.0:
        raise ValueError("Guided-filter epsilon and edge power must be positive.")
    mask = np.asarray(confidence, dtype=np.float32).squeeze()
    if mask.ndim != 2:
        raise ValueError(f"Expected a 2-D confidence mask, got {mask.shape}.")
    height, width = output_shape
    if mask.shape != (height, width):
        mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_LINEAR)
    if guidance_rgb is not None:
        mask = _guided_filter(mask, guidance_rgb, guided_radius, guided_epsilon)
    # Explicit trimap: certain background=0, certain foreground=1, only the
    # uncertain contour receives a soft transition.
    mask = _smoothstep((mask - threshold_low) / (threshold_high - threshold_low))
    if close_radius:
        size = 2 * close_radius + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    if dilate_radius:
        size = 2 * dilate_radius + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        mask = cv2.dilate(mask, kernel)
    if feather:
        mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=feather, sigmaY=feather)
    mask = np.power(np.clip(mask, 0.0, 1.0), edge_power)
    mask[mask < 2.0e-2] = 0.0
    mask[mask > 1.0 - 2.0e-2] = 1.0
    return mask[..., None]


class MaskEMA:
    """Suppress frame-to-frame mask shimmer while retaining soft boundaries."""

    def __init__(self, beta: float = 0.25) -> None:
        if not 0.0 <= beta < 1.0:
            raise ValueError("mask beta must be in [0, 1).")
        self.beta = beta
        self.state: np.ndarray | None = None

    def update(self, mask: np.ndarray) -> np.ndarray:
        value = np.asarray(mask, dtype=np.float32)
        if self.state is None or self.state.shape != value.shape:
            self.state = value.copy()
        else:
            self.state *= self.beta
            self.state += (1.0 - self.beta) * value
        result = np.clip(self.state, 0.0, 1.0).copy()
        result[result < 2.0e-2] = 0.0
        result[result > 1.0 - 2.0e-2] = 1.0
        return result


class MotionAlignedMaskEMA:
    """Warp the previous mask to the current frame before temporal smoothing."""

    def __init__(self, beta: float = 0.80, flow_width: int = 256) -> None:
        if not 0.0 <= beta < 1.0:
            raise ValueError("motion mask beta must be in [0, 1).")
        if flow_width < 64:
            raise ValueError("flow_width must be at least 64 pixels.")
        self.beta = beta
        self.flow_width = flow_width
        self.previous_gray: np.ndarray | None = None
        self.state: np.ndarray | None = None
        self.flow = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST)

    def _small_gray(self, rgb: np.ndarray) -> np.ndarray:
        height, width = rgb.shape[:2]
        target_height = max(1, round(height * self.flow_width / width))
        small = cv2.resize(rgb, (self.flow_width, target_height),
                           interpolation=cv2.INTER_AREA)
        return cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)

    def update(self, mask: np.ndarray, rgb: np.ndarray) -> np.ndarray:
        height, width = mask.shape[:2]
        gray = self._small_gray(rgb)
        current = cv2.resize(mask[..., 0], (gray.shape[1], gray.shape[0]),
                             interpolation=cv2.INTER_AREA)
        if self.state is None or self.previous_gray is None or self.state.shape != current.shape:
            fused = current
        else:
            # Backward flow maps each current-frame location into the previous frame.
            backward = self.flow.calc(gray, self.previous_gray, None)
            yy, xx = np.mgrid[0:gray.shape[0], 0:gray.shape[1]].astype(np.float32)
            warped = cv2.remap(
                self.state, xx + backward[..., 0], yy + backward[..., 1],
                interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                borderValue=0.0,
            )
            fused = self.beta * warped + (1.0 - self.beta) * current
            # Do not let temporal smoothing invent foreground where both masks
            # confidently say background, or erase confident new foreground.
            fused[(current < 0.02) & (warped < 0.02)] = 0.0
            fused[current > 0.98] = np.maximum(fused[current > 0.98], 0.98)
        self.previous_gray = gray
        self.state = np.clip(fused, 0.0, 1.0).astype(np.float32)
        result = cv2.resize(self.state, (width, height), interpolation=cv2.INTER_LINEAR)
        result[result < 0.02] = 0.0
        result[result > 0.98] = 1.0
        return result[..., None]


def suppress_boundary_gain(person_mask: np.ndarray,
                           fade_pixels: float = 8.0) -> np.ndarray:
    """Make relighting approach identity at the silhouette to prevent halos."""
    if fade_pixels < 0.0:
        raise ValueError("fade_pixels must be non-negative.")
    if fade_pixels == 0.0:
        return person_mask.copy()
    alpha = np.clip(person_mask[..., 0].astype(np.float32), 0.0, 1.0)
    support = np.uint8(alpha > 0.02)
    distance = cv2.distanceTransform(support, cv2.DIST_L2, 5)
    # OpenCV reports a distance of roughly one pixel on the first foreground
    # pixel; subtract it so the visible silhouette itself remains unchanged.
    ramp = _smoothstep(np.maximum(distance - 1.0, 0.0) /
                       max(fade_pixels, 1.0e-6))
    result = alpha * ramp
    result[result < 0.02] = 0.0
    result[result > 0.98] = 1.0
    return result[..., None]


def composite_person(source_rgb: np.ndarray, enhanced_rgb: np.ndarray,
                     person_mask: np.ndarray) -> np.ndarray:
    """Use RRNet only inside the person mask and preserve the source elsewhere."""
    if source_rgb.shape != enhanced_rgb.shape:
        raise ValueError("Source and enhanced frames must have identical shapes.")
    if person_mask.shape != source_rgb.shape[:2] + (1,):
        raise ValueError("person_mask must have shape [H, W, 1].")
    source = source_rgb.astype(np.float32)
    enhanced = enhanced_rgb.astype(np.float32)
    alpha = np.clip(person_mask.astype(np.float32), 0.0, 1.0)
    blended = source + alpha * (enhanced - source)
    return np.uint8(np.clip(blended + 0.5, 0.0, 255.0))


def composite_person_tensor(source: torch.Tensor, enhanced: torch.Tensor,
                            person_mask: torch.Tensor) -> torch.Tensor:
    """Blend a relit person on CUDA without a full-resolution NumPy round trip."""
    if source.shape != enhanced.shape or source.ndim != 4:
        raise ValueError("source and enhanced must have identical [B,C,H,W] shapes.")
    if person_mask.shape != (source.shape[0], 1, source.shape[2], source.shape[3]):
        raise ValueError("person_mask must have shape [B,1,H,W].")
    alpha = person_mask.to(device=source.device, dtype=source.dtype).clamp(0.0, 1.0)
    return source + alpha * (enhanced - source)


def _smoothstep_tensor(value: torch.Tensor) -> torch.Tensor:
    value = value.clamp(0.0, 1.0)
    return value * value * (3.0 - 2.0 * value)


def _guided_filter_tensor(confidence: torch.Tensor, guidance: torch.Tensor,
                          radius: int, epsilon: float) -> torch.Tensor:
    """Torch equivalent of the single-channel guided filter used by the CPU path."""
    if radius <= 0:
        return confidence
    kernel = 2 * radius + 1

    def mean(value: torch.Tensor) -> torch.Tensor:
        return F.avg_pool2d(value, kernel, stride=1, padding=radius,
                            count_include_pad=False)

    mean_i = mean(guidance)
    mean_p = mean(confidence)
    variance_i = mean(guidance * guidance) - mean_i * mean_i
    covariance_ip = mean(guidance * confidence) - mean_i * mean_p
    a = covariance_ip / (variance_i + epsilon)
    b = mean_p - a * mean_i
    return (mean(a) * guidance + mean(b)).clamp(0.0, 1.0)


def _gaussian_blur_tensor(value: torch.Tensor, sigma: float) -> torch.Tensor:
    if sigma <= 0.0:
        return value
    radius = max(1, int(round(3.0 * sigma)))
    coordinates = torch.arange(
        -radius, radius + 1, device=value.device, dtype=value.dtype)
    kernel = torch.exp(-(coordinates * coordinates) / (2.0 * sigma * sigma))
    kernel /= kernel.sum()
    horizontal = kernel.view(1, 1, 1, -1)
    vertical = kernel.view(1, 1, -1, 1)
    value = F.pad(value, (radius, radius, 0, 0), mode="replicate")
    value = F.conv2d(value, horizontal)
    value = F.pad(value, (0, 0, radius, radius), mode="replicate")
    return F.conv2d(value, vertical)


def _morphological_close_tensor(value: torch.Tensor, radius: int) -> torch.Tensor:
    if radius <= 0:
        return value
    kernel = 2 * radius + 1
    dilated = F.max_pool2d(value, kernel, stride=1, padding=radius)
    return -F.max_pool2d(-dilated, kernel, stride=1, padding=radius)


def _boundary_fade_tensor(mask: torch.Tensor, fade_pixels: float) -> torch.Tensor:
    """Approximate the CPU distance-transform fade using small GPU erosions."""
    if fade_pixels <= 0.0:
        return mask
    steps = max(1, int(round(fade_pixels)))
    current = (mask > 0.02).to(mask.dtype)
    distance = torch.zeros_like(mask)
    for _ in range(steps + 1):
        distance = distance + current
        current = -F.max_pool2d(-current, 3, stride=1, padding=1)
    ramp = _smoothstep_tensor((distance - 1.0).clamp_min(0.0) / max(float(steps), 1.0))
    return mask * ramp


class MediaPipeRawPersonSegmenter:
    """MediaPipe VIDEO-mode inference without expensive full-resolution CPU refinement."""

    def __init__(self, model_path: str | Path) -> None:
        try:
            import mediapipe as mp
        except ImportError as error:
            raise RuntimeError(
                "Person masking requires mediapipe. Install the project requirements."
            ) from error
        model_path = Path(model_path).resolve()
        if not model_path.is_file():
            raise FileNotFoundError(f"Person segmentation model not found: {model_path}")
        self.mp = mp
        options = mp.tasks.vision.ImageSegmenterOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            output_confidence_masks=True,
            output_category_mask=False,
        )
        self.segmenter = mp.tasks.vision.ImageSegmenter.create_from_options(options)

    def segment(self, rgb: np.ndarray, timestamp_ms: int) -> np.ndarray:
        if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError("MediaPipe input must be an RGB uint8 image [H,W,3].")
        image = self.mp.Image(
            image_format=self.mp.ImageFormat.SRGB,
            data=np.ascontiguousarray(rgb),
        )
        result = self.segmenter.segment_for_video(image, timestamp_ms)
        if not result.confidence_masks:
            raise RuntimeError("MediaPipe returned no person confidence mask.")
        return np.asarray(result.confidence_masks[0].numpy_view()).copy()

    def close(self) -> None:
        self.segmenter.close()


class CudaPersonMasker:
    """Sparse MediaPipe inference with CUDA refinement and cached intermediate masks."""

    def __init__(self, model_path: str | Path, *, device: torch.device,
                 mask_every: int = 3, work_width: int = 512,
                 threshold_low: float = 0.30, threshold_high: float = 0.70,
                 close_radius: int = 2, dilate_radius: int = 0,
                 feather: float = 1.5, temporal_beta: float = 0.80,
                 guided_radius: int = 5, guided_epsilon: float = 1.0e-3,
                 edge_power: float = 1.5, boundary_fade: float = 8.0) -> None:
        if device.type != "cuda":
            raise ValueError("CudaPersonMasker requires a CUDA device.")
        if mask_every < 1 or work_width < 64:
            raise ValueError("mask_every must be >=1 and work_width must be >=64.")
        if not 0.0 <= threshold_low < threshold_high <= 1.0:
            raise ValueError("Mask thresholds must satisfy 0 <= low < high <= 1.")
        self.raw_segmenter = MediaPipeRawPersonSegmenter(model_path)
        self.device = device
        self.mask_every = mask_every
        self.work_width = work_width
        self.threshold_low = threshold_low
        self.threshold_high = threshold_high
        self.close_radius = close_radius
        self.dilate_radius = dilate_radius
        self.feather = feather
        self.temporal_beta = temporal_beta
        self.guided_radius = guided_radius
        self.guided_epsilon = guided_epsilon
        self.edge_power = edge_power
        self.boundary_fade = boundary_fade
        self.state: torch.Tensor | None = None

    def _work_shape(self, height: int, width: int) -> tuple[int, int]:
        work_width = min(self.work_width, width)
        work_height = max(1, round(height * work_width / width))
        return work_height, work_width

    def segment(self, rgb: np.ndarray, timestamp_ms: int, frame_index: int,
                guidance: torch.Tensor) -> torch.Tensor:
        height, width = rgb.shape[:2]
        work_height, work_width = self._work_shape(height, width)
        update = self.state is None or frame_index % self.mask_every == 0
        if update:
            confidence = self.raw_segmenter.segment(rgb, timestamp_ms)
            value = torch.from_numpy(confidence).to(
                device=self.device, dtype=torch.float32).squeeze().view(1, 1, *confidence.squeeze().shape)
            value = F.interpolate(value, (work_height, work_width), mode="bilinear",
                                  align_corners=False)
            gray = (0.299 * guidance[:, 0:1] + 0.587 * guidance[:, 1:2]
                    + 0.114 * guidance[:, 2:3]).float()
            gray = F.interpolate(gray, (work_height, work_width), mode="bilinear",
                                 align_corners=False)
            scale = work_width / max(width, 1)
            guided_radius = max(1, round(self.guided_radius * scale))
            value = _guided_filter_tensor(
                value, gray, guided_radius, self.guided_epsilon)
            value = _smoothstep_tensor(
                (value - self.threshold_low) /
                (self.threshold_high - self.threshold_low))
            close_radius = max(0, round(self.close_radius * scale))
            dilate_radius = max(0, round(self.dilate_radius * scale))
            value = _morphological_close_tensor(value, close_radius)
            if dilate_radius:
                kernel = 2 * dilate_radius + 1
                value = F.max_pool2d(value, kernel, stride=1, padding=dilate_radius)
            feather = self.feather * scale
            if feather >= 0.35:
                value = _gaussian_blur_tensor(value, feather)
            value = value.clamp(0.0, 1.0).pow(self.edge_power)
            if self.state is not None and self.state.shape == value.shape:
                # Preserve approximately the same smoothing per unit time when
                # the neural segmenter runs once every mask_every frames.
                beta = self.temporal_beta ** self.mask_every
                value = beta * self.state + (1.0 - beta) * value
            fade = self.boundary_fade * scale
            value = _boundary_fade_tensor(value, fade)
            self.state = value.detach()

        result = F.interpolate(
            self.state, (height, width), mode="bilinear", align_corners=False)
        result = result.clamp(0.0, 1.0)
        result = torch.where(result < 0.02, torch.zeros_like(result), result)
        return torch.where(result > 0.98, torch.ones_like(result), result)

    def close(self) -> None:
        self.raw_segmenter.close()


class MediaPipePersonMasker:
    """MediaPipe Tasks VIDEO-mode wrapper returning a refined full-person mask."""

    def __init__(self, model_path: str | Path, *, threshold_low: float = 0.30,
                 threshold_high: float = 0.70, close_radius: int = 2,
                 dilate_radius: int = 0, feather: float = 1.5,
                 temporal_beta: float = 0.80, guided_radius: int = 5,
                 guided_epsilon: float = 1.0e-3,
                 edge_power: float = 1.5, flow_width: int = 256,
                 boundary_fade: float = 8.0) -> None:
        try:
            import mediapipe as mp
        except ImportError as error:
            raise RuntimeError(
                "Person masking requires mediapipe. Install the project requirements."
            ) from error
        model_path = Path(model_path).resolve()
        if not model_path.is_file():
            raise FileNotFoundError(f"Person segmentation model not found: {model_path}")
        self.mp = mp
        options = mp.tasks.vision.ImageSegmenterOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            output_confidence_masks=True,
            output_category_mask=False,
        )
        self.segmenter = mp.tasks.vision.ImageSegmenter.create_from_options(options)
        self.output_options = {
            "threshold_low": threshold_low,
            "threshold_high": threshold_high,
            "close_radius": close_radius,
            "dilate_radius": dilate_radius,
            "feather": feather,
            "guided_radius": guided_radius,
            "guided_epsilon": guided_epsilon,
            "edge_power": edge_power,
        }
        self.temporal = MotionAlignedMaskEMA(temporal_beta, flow_width)
        self.boundary_fade = boundary_fade

    def segment(self, rgb: np.ndarray, timestamp_ms: int) -> np.ndarray:
        if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError("MediaPipe input must be an RGB uint8 image [H,W,3].")
        image = self.mp.Image(
            image_format=self.mp.ImageFormat.SRGB,
            data=np.ascontiguousarray(rgb),
        )
        result = self.segmenter.segment_for_video(image, timestamp_ms)
        if not result.confidence_masks:
            raise RuntimeError("MediaPipe returned no person confidence mask.")
        confidence = np.asarray(result.confidence_masks[0].numpy_view()).copy()
        mask = refine_person_mask(
            confidence, rgb.shape[:2], guidance_rgb=rgb, **self.output_options)
        mask = self.temporal.update(mask, rgb)
        return suppress_boundary_gain(mask, self.boundary_fade)

    def close(self) -> None:
        self.segmenter.close()
