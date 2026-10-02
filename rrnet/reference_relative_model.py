"""RRNet reference relighting through explicit relative illumination transfer."""

from __future__ import annotations

import torch
from torch import nn

from .losses import luminance
from .model import RRNet


class ReferenceRelativeRRNet(nn.Module):
    """Estimate input/reference light separately and transfer their ratio.

    Both images share the same LPRM.  The reference light is re-rendered on the
    source depth before division, so identity, texture, and geometry always come
    from the source participant.  No dense image residual is used.
    """

    def __init__(self, *, illumination_floor: float = 0.05,
                 min_transfer_gain: float = 0.40,
                 max_transfer_gain: float = 3.00,
                 achromatic_illumination: bool = True,
                 preserve_chromaticity: bool = True,
                 output_headroom: float = 0.999,
                 **rrnet_kwargs) -> None:
        super().__init__()
        if illumination_floor <= 0.0:
            raise ValueError("illumination_floor must be positive")
        if not 0.0 < min_transfer_gain <= 1.0:
            raise ValueError("min_transfer_gain must be in (0, 1]")
        if max_transfer_gain < 1.0:
            raise ValueError("max_transfer_gain must be at least one")
        if min_transfer_gain > max_transfer_gain:
            raise ValueError("min_transfer_gain cannot exceed max_transfer_gain")
        if not 0.0 < output_headroom <= 1.0:
            raise ValueError("output_headroom must be in (0, 1]")
        self.base = RRNet(**rrnet_kwargs)
        if self.base.agm is not None:
            raise ValueError("ReferenceRelativeRRNet currently requires use_agm: false")
        self.illumination_floor = float(illumination_floor)
        self.min_transfer_gain = float(min_transfer_gain)
        self.max_transfer_gain = float(max_transfer_gain)
        self.achromatic_illumination = bool(achromatic_illumination)
        self.preserve_chromaticity = bool(preserve_chromaticity)
        self.output_headroom = float(output_headroom)

    @property
    def agm(self):
        return self.base.agm

    @property
    def depth(self):
        return self.base.depth

    @property
    def renderer(self):
        return self.base.renderer

    def train(self, mode: bool = True) -> "ReferenceRelativeRRNet":
        super().train(mode)
        self.base.depth.eval()
        return self

    @staticmethod
    def prepare_light_input(image: torch.Tensor,
                            mask: torch.Tensor | None) -> torch.Tensor:
        """Remove identity colour and ignored background before light encoding."""
        gray = luminance(image)
        if mask is None:
            return gray.expand(-1, 3, -1, -1)
        mask = mask.to(device=gray.device, dtype=gray.dtype).clamp(0.0, 1.0)
        valid = mask.sum(dim=(2, 3), keepdim=True).clamp_min(1.0)
        mean = (gray * mask).sum(dim=(2, 3), keepdim=True) / valid
        prepared = gray * mask + mean * (1.0 - mask)
        return prepared.expand(-1, 3, -1, -1)

    def estimate_light(self, image: torch.Tensor,
                       mask: torch.Tensor | None = None) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        return self.base.lprm(self.prepare_light_input(image, mask))

    def encode_reference(self, reference: torch.Tensor,
                         reference_mask: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        prediction = self.estimate_light(reference, reference_mask)
        return {
            "theta": prediction["theta"],
            "theta_normalized": prediction["theta_normalized"],
        }

    def illumination_from_theta(self, depth: torch.Tensor,
                                theta: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        illumination, normals = self.renderer.illumination(depth, theta)
        if self.achromatic_illumination:
            illumination = luminance(illumination).expand(-1, 3, -1, -1)
        return illumination, normals

    def transfer(self, source: torch.Tensor, source_depth: torch.Tensor,
                 source_theta: torch.Tensor,
                 reference_theta: torch.Tensor,
                 relight_mask: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        source_light, normals = self.illumination_from_theta(source_depth, source_theta)
        reference_light, _ = self.illumination_from_theta(source_depth, reference_theta)
        transfer_gain = self.compute_transfer_gain(
            source, source_light, reference_light)
        raw_output = source * transfer_gain
        output = (raw_output.clamp(0.0, 1.0)
                  if self.renderer.clamp_output else raw_output)
        result = {
            "output": output,
            "raw_output": raw_output,
            "depth": source_depth,
            "theta": reference_theta,
            "source_theta": source_theta,
            "reference_theta": reference_theta,
            "source_illumination": source_light,
            "reference_illumination_on_source": reference_light,
            "transfer_gain": transfer_gain,
            "normals": normals,
        }
        if relight_mask is not None:
            result["loss_output"] = self.renderer.blend_relight(
                raw_output, source, relight_mask)
            result["output"] = self.renderer.blend_relight(
                output, source, relight_mask)
        return result

    def compute_transfer_gain(self, source: torch.Tensor,
                              source_light: torch.Tensor,
                              reference_light: torch.Tensor) -> torch.Tensor:
        """Return a bounded shared RGB gain without channel clipping.

        A common gain preserves per-pixel chromaticity only while no individual
        channel clips.  The optional headroom cap enforces that condition.
        """
        denominator = source_light.clamp_min(self.illumination_floor)
        gain = (reference_light.clamp_min(0.0) / denominator).clamp(
            self.min_transfer_gain, self.max_transfer_gain)
        if self.preserve_chromaticity:
            source_peak = source.clamp_min(0.0).amax(
                dim=1, keepdim=True).clamp_min(1.0e-4)
            chroma_safe_gain = self.output_headroom / source_peak
            gain = torch.minimum(gain, chroma_safe_gain)
        return gain

    def forward(self, source: torch.Tensor, reference: torch.Tensor,
                relight_mask: torch.Tensor | None = None,
                reference_mask: torch.Tensor | None = None,
                source_light_mask: torch.Tensor | None = None,
                source_depth_override: torch.Tensor | None = None,
                reference_depth_override: torch.Tensor | None = None,
                ) -> dict[str, torch.Tensor]:
        source_prediction = self.estimate_light(source, source_light_mask)
        reference_prediction = self.estimate_light(reference, reference_mask)
        with torch.no_grad():
            source_depth = (self.depth(source) if source_depth_override is None
                            else source_depth_override)
            reference_depth = (
                self.depth(reference) if reference_depth_override is None
                else reference_depth_override)
        result = self.transfer(
            source,
            source_depth,
            source_prediction["theta"],
            reference_prediction["theta"],
            relight_mask,
        )
        reference_light, _ = self.illumination_from_theta(
            reference_depth, reference_prediction["theta"])
        result.update({
            "reference_depth": reference_depth,
            "reference_illumination": reference_light,
            "source_theta_normalized": source_prediction["theta_normalized"],
            "reference_theta_normalized": reference_prediction["theta_normalized"],
            "source_theta0": source_prediction["theta0"],
            "source_theta_offset": source_prediction["theta_offset"],
            "reference_theta0": reference_prediction["theta0"],
            "reference_theta_offset": reference_prediction["theta_offset"],
        })
        return result

    @staticmethod
    def _expand_group(value: torch.Tensor, group_size: int) -> torch.Tensor:
        """Repeat one value per group in contiguous group-member order."""
        return value.unsqueeze(1).expand(
            value.shape[0], group_size, *value.shape[1:]
        ).reshape(-1, *value.shape[1:])

    def forward_grouped(
            self, source: torch.Tensor, reference: torch.Tensor,
            group_size: int,
            relight_mask: torch.Tensor | None = None,
            reference_mask: torch.Tensor | None = None,
            source_light_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Train several source lights against one reference without duplication.

        The dataset/collator repeats common reference tensors so old data loading
        remains compatible. Here only one reference per group is encoded. Source
        variants and unique references are sent through LPRM in one joint call,
        which prevents repeated references from corrupting BatchNorm statistics.
        """
        if group_size < 2 or source.shape[0] % group_size:
            raise ValueError(
                f"Invalid grouped batch: batch={source.shape[0]}, "
                f"group_size={group_size}."
            )
        reference_unique = reference[::group_size]
        reference_mask_unique = (
            reference_mask[::group_size] if reference_mask is not None else None)
        prepared_source = self.prepare_light_input(source, source_light_mask)
        prepared_reference = self.prepare_light_input(
            reference_unique, reference_mask_unique)
        source_count = source.shape[0]
        joint_prediction = self.base.lprm(torch.cat(
            (prepared_source, prepared_reference), dim=0))

        source_theta = joint_prediction["theta"][:source_count]
        reference_theta_unique = joint_prediction["theta"][source_count:]
        reference_theta = self._expand_group(
            reference_theta_unique, group_size)
        with torch.no_grad():
            source_depth = self.depth(source)
            reference_depth_unique = self.depth(reference_unique)
        result = self.transfer(
            source, source_depth, source_theta, reference_theta, relight_mask)

        reference_light_unique, _ = self.illumination_from_theta(
            reference_depth_unique, reference_theta_unique)
        result.update({
            "reference_depth": self._expand_group(
                reference_depth_unique, group_size),
            "reference_illumination": self._expand_group(
                reference_light_unique, group_size),
            "source_theta_normalized": joint_prediction[
                "theta_normalized"][:source_count],
            "reference_theta_normalized": self._expand_group(
                joint_prediction["theta_normalized"][source_count:], group_size),
            "source_theta0": joint_prediction["theta0"][:source_count],
            "source_theta_offset": joint_prediction[
                "theta_offset"][:source_count],
            "reference_theta0": self._expand_group(
                joint_prediction["theta0"][source_count:], group_size),
            "reference_theta_offset": self._expand_group(
                joint_prediction["theta_offset"][source_count:], group_size),
        })
        return result

    def forward_physical_grouped(
            self, source: torch.Tensor, reference: torch.Tensor,
            group_size: int, uniform_group_mask: torch.Tensor,
            relight_mask: torch.Tensor | None = None,
            reference_mask: torch.Tensor | None = None,
            source_light_mask: torch.Tensor | None = None,
            source_depth_override: torch.Tensor | None = None,
            reference_depth_override: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Encode only unique images in mixed physical training groups.

        Uniformity groups contain K distinct sources and one repeated reference;
        reference-sensitivity groups contain one repeated source and K distinct
        references. Removing those exact duplicates before LPRM is essential:
        otherwise repeated images corrupt BatchNorm running statistics and create
        a large train/eval gap.
        """
        if group_size < 2 or source.shape[0] % group_size:
            raise ValueError("Invalid physical grouped batch")
        flags = uniform_group_mask.reshape(-1, group_size)
        if not torch.all(flags == flags[:, :1]):
            raise ValueError("uniform_group_mask must be constant within a group")

        source_unique: list[int] = []
        reference_unique: list[int] = []
        source_map: list[int] = []
        reference_map: list[int] = []
        for group_index, is_uniform in enumerate(flags[:, 0].tolist()):
            start = group_index * group_size
            members = list(range(start, start + group_size))
            if is_uniform:
                source_positions = members
                reference_positions = [start]
            else:
                source_positions = [start]
                reference_positions = members
            source_offset = len(source_unique)
            reference_offset = len(reference_unique)
            source_unique.extend(source_positions)
            reference_unique.extend(reference_positions)
            source_map.extend(
                [source_offset + member if is_uniform else source_offset
                 for member in range(group_size)])
            reference_map.extend(
                [reference_offset if is_uniform else reference_offset + member
                 for member in range(group_size)])

        source_indices = torch.tensor(source_unique, device=source.device)
        reference_indices = torch.tensor(reference_unique, device=source.device)
        source_expand = torch.tensor(source_map, device=source.device)
        reference_expand = torch.tensor(reference_map, device=source.device)
        prepared_source = self.prepare_light_input(
            source.index_select(0, source_indices),
            source_light_mask.index_select(0, source_indices)
            if source_light_mask is not None else None)
        prepared_reference = self.prepare_light_input(
            reference.index_select(0, reference_indices),
            reference_mask.index_select(0, reference_indices)
            if reference_mask is not None else None)
        source_unique_count = len(source_unique)
        joint = self.base.lprm(torch.cat((prepared_source, prepared_reference), dim=0))

        def expanded(key: str) -> tuple[torch.Tensor, torch.Tensor]:
            source_value = joint[key][:source_unique_count].index_select(
                0, source_expand)
            reference_value = joint[key][source_unique_count:].index_select(
                0, reference_expand)
            return source_value, reference_value

        source_theta, reference_theta = expanded("theta")
        source_theta_normalized, reference_theta_normalized = expanded(
            "theta_normalized")
        source_theta0, reference_theta0 = expanded("theta0")
        source_theta_offset, reference_theta_offset = expanded("theta_offset")
        with torch.no_grad():
            source_depth = (self.depth(source) if source_depth_override is None
                            else source_depth_override)
            reference_depth = (
                self.depth(reference) if reference_depth_override is None
                else reference_depth_override)
        result = self.transfer(
            source, source_depth, source_theta, reference_theta, relight_mask)
        reference_light, _ = self.illumination_from_theta(
            reference_depth, reference_theta)
        result.update({
            "reference_depth": reference_depth,
            "reference_illumination": reference_light,
            "source_theta_normalized": source_theta_normalized,
            "reference_theta_normalized": reference_theta_normalized,
            "source_theta0": source_theta0,
            "source_theta_offset": source_theta_offset,
            "reference_theta0": reference_theta0,
            "reference_theta_offset": reference_theta_offset,
        })
        return result
