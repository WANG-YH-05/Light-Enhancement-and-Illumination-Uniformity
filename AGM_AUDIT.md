# RRNet AGM implementation audit

This note separates what the RRNet paper explicitly specifies from assumptions
in this repository. It applies to the paper *RRNet: Configurable Real-Time Video
Enhancement with Arbitrary Local Lighting Variations* (arXiv:2601.01865) and the
current implementation in `rrnet/agm.py`.

## Bottom line

The current AGM is a reasonable implementation of the paper's **high-level
description**, but it is not a verifiable exact reproduction. The paper omits
enough architectural and training details that several parts of the module have
to be guessed. It must therefore not be described as the official AGM.

More importantly, the paper's AGM is an optional albedo estimator for canonical
enhancement and glare removal. It is not a reference-light transfer module. The
current `ReferenceRelativeRRNet` intentionally rejects `use_agm: true`; adding
AGM to that path would be a new method rather than a faithful RRNet feature.

## What the paper explicitly defines

Section III-B and Figure 3 state that AGM:

- is optional and is intended for difficult static-image cases such as glare;
- receives features from the full-resolution LPRM encoder branch;
- uses a lightweight U-Net-style decoder;
- contains one residual block and bilinear upsampling;
- predicts a three-channel illumination mask `Z'`;
- upsamples `Z'` to `Z` and computes albedo as `A = I - Z`;
- replaces image `I` with albedo `A` as the base input to the rendering module.

The paper's complete stated training loss remains:

`L_total = L_pixel + L_roi + lambda_r * L_reg`

It does not specify a separate albedo, illumination-mask, chromaticity, temporal,
or identity loss for AGM.

Table II reports the following FFHQL ablation on an RTX 3090:

| Variant | NIQE (lower) | FID | Runtime |
| --- | ---: | ---: | ---: |
| Baseline: dual branch + AGM + 9 lights | 3.71 | 23.05 | 17.0 ms |
| Remove AGM | 4.07 | 23.24 | 15.7 ms |

This supports AGM for the paper's ordinary enhancement benchmark. It does not
test arbitrary reference-person light transfer.

## What the current code matches

`rrnet/agm.py` matches the stated high-level operations:

- it consumes multiscale features from the refined/full-resolution encoder pass;
- it has one residual block;
- it upsamples with bilinear interpolation;
- it emits an unconstrained three-channel `Z'` and an upsampled `Z`;
- it computes `albedo = image - z`;
- `rrnet/model.py` passes that albedo to the depth-aware renderer.

The existing implementation is therefore suitable as a research approximation
for ablation, provided that reports call it a reconstruction rather than an
official implementation.

## Details that are not recoverable from the paper

The following choices in this repository are assumptions:

- decoder width (`agm_channels`, currently commonly 64);
- exact number and placement of U-Net skip connections;
- convolution kernel sizes and activation functions;
- whether normalization is used in the decoder;
- whether `Z'`, `Z`, or `A` is bounded or normalized;
- whether the encoder starts from ImageNet/RepViT pretrained weights;
- initialization of the final AGM layer;
- exact values of `sigma1`, `sigma2`, ROI constants, and regularization weights;
- any additional unpublished stabilization used for albedo/light separation;
- how AGM output is handled temporally in video.

No official RRNet repository, pretrained weights, FFHQL release, or supplemental
implementation was publicly discoverable at the time of this audit.

## Important mismatch with the current project goal

The project's best reference-relative model estimates source and reference light,
renders both on source geometry, and transfers their ratio. That method is an
extension beyond the paper. It also prepares the light-regression input as masked
grayscale and enforces achromatic illumination, neither of which is the paper's
canonical RGB RRNet path.

AGM cannot simply be switched on in this model:

1. The paper predicts albedo only for the current input image.
2. It does not define how source albedo should interact with a separately inferred
   reference light.
3. It does not define temporal smoothing for the dense three-channel AGM mask.
4. Its three-channel mask can alter local chromaticity, conflicting with the
   project's requirement not to change perceived skin colour or ethnicity.

For these reasons, `ReferenceRelativeRRNet` rejects `use_agm: true` rather than
silently running an undefined combination.

## Interpretation of earlier artifacts

The earlier red/blue patches and unstable hair boundaries cannot be attributed
to one component without a same-checkpoint ablation. The project has used both a
custom Delta-Y residual branch and RGB lighting variants at different points.
The current evidence supports only these narrower conclusions:

- the old Delta-Y reference branch produced visible spatial and temporal artifacts;
- a free three-channel AGM is also capable of changing local colour because `Z`
  has independent RGB channels;
- the paper smooths only lighting parameters `theta`; it gives no smoothing rule
  for dense AGM masks, so video stability is not guaranteed by Equation (5).

## Engineering decision

- Keep the current stable reference-relative model AGM-free.
- Do not claim that enabling the existing AGM will solve reference-light
  uniformity.
- Keep AGM only as a separately labelled canonical-RRNet reconstruction and
  research ablation.
- Do not start another large AGM training run until there is either official
  implementation detail or a clearly defined new, colour-safe, temporally stable
  extension with its own name and evaluation.

