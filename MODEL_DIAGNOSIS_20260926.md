# RRNet reference-relighting project: diagnosis and decision, 2026-09-26

## Decision

Keep the project and the existing stable model as a baseline. Stop extending the
current AGM A/B training unchanged. The evidence does not justify abandoning
AGM as a class, or assuming that another large training run will fix it.
First resolve concrete implementation/modeling issues and run a bounded
representational and small-set learning experiment. Add an official pretrained
DPR comparison before committing to a new main architecture.

No model/configuration was modified for this audit and no training was started.
The only GPU diagnostic was frozen depth inference on one validation clean frame
per person (11, 14, 37), plus a synthetic plane normal calculation.

## Verified findings

### 1. Depth/normal coordinate scales are inconsistent

`rrnet/depth.py:20` differentiates normalized depth per pixel, with z-component
fixed at 1. `rrnet/renderer.py` constructs x/y coordinates on [0,1]. Derivatives
do not account for the corresponding x/y spacing. As implemented, a unit ramp
across the image tilts by 0.224688 degrees at 256 pixels and 0.112124 degrees at
512 pixels. Under a common normalized orthographic coordinate convention, that
same unit ramp should have the same normal at both resolutions (45 degrees
if depth and horizontal coordinate have the same unit).

Measured on the first validation clean frame for each person, at 256x256:

| Person | Median skin normal tilt | 95th percentile |
| --- | ---: | ---: |
| MEAD_front_11 | 0.3021 degrees | 1.9484 degrees |
| MEAD_front_14 | 0.3367 degrees | 1.8817 degrees |
| MEAD_front_37 | 0.2896 degrees | 1.1967 degrees |

This suppresses facial orientation effects in N dot light_direction. The distance
attenuation term still varies spatially, so this does NOT mean that all virtual
lights are mathematically global exposure scalars. It is a verified numerical
issue; its share of the final visual failure remains to be measured.

Choose and document geometry coordinates/depth scale, then verify plane/sphere
and resolution consistency. Relative Depth Anything outputs do not become metric
geometry simply by multiplying derivatives by image width. Changed geometry must
be evaluated in a new experiment: old trained lighting parameters may depend on
the old renderer.

### 2. RGB additive AGM receives grayscale-only learned features

`train_agm_physical.py:113` calls `prepare_light_input`, which converts source
RGB to luminance and replaces the background with its masked mean. The AGM
decoder computes Z from these features alone. RGB is used only in A = I - Z;
there is no direct RGB feature branch for predicting Z.

For achromatic multiplicative degradation I = C*S, the correct additive
correction is Z = I-C = I*(1-1/S), which depends on local RGB. Equal-luminance
images can require different RGB corrections. This is an information bottleneck
for a general RGB residual predictor, even though grayscale features can still
support useful brightness correction. It does not alone explain all observed
under-brightening.

The encoder was adapted for a previous light-regression task. The A/B encoder
branch uses learning rate 1e-6 and keeps BatchNorm statistics frozen. Its failure
to outperform the frozen branch does not prove that a properly conditioned or
trained encoder cannot help.

### 3. Clean MEAD is not measured albedo

If C = A_real*S_original, current synthetic inputs are C*S_added. Supervision
back to C removes S_added, while preserving S_original. The re-rendered target
C*S_target also retains S_original. This exact synthetic pairing is useful for
diagnosis, but cannot establish removal of arbitrary real capture lighting or
exact light equivalence across different MEAD participants.

The current physical sampler uses achromatic lights and has no explicit cast
shadow visibility or specular BRDF. Rendering is applied to loaded image values
without a linear-light conversion. It does not cover the full range of real
camera processing, colored lighting, noise, clipping and specular highlights.

### 4. Known-target rendering loss is not strictly necessary for base recovery

Inside the foreground, without clipping, output = A_hat*S_target and target =
C*S_target. Their L1 difference is |S_target|*|A_hat-C|. Thus it is largely a
target-dependent weighting of base reconstruction. It does not teach reference
image estimation, because the AGM is never conditioned on a reference image or
target theta. Earlier statements that this loss was a necessary step were too
strong. It can be kept as a consistency diagnostic or auxiliary objective.

The pilot has shared targets across groups but no explicit pairwise group loss.
Shared ground truth already provides an implicit consistency signal. Adding a
pairwise term alone will not resolve poor base recovery or missing information.

### 5. Soft-mask composition introduces a small target/base conflict

Let mask=m, clean=C, source-light=S, target-light=T. Synthetic source is
(1-m)C+mCS and target is (1-m)C+mCT. Even if AGM predicts C exactly, blending
the rendered base with source yields target + m(1-m)C(S-1).
This affects feathered boundaries. Existing known-light inverse reports show
small errors here, so it is not established as the dominant whole-face failure.
Use one consistent compositing convention for generated targets and predictions.

## What the experiments actually establish

Both 9909-step branches complete one pass through 19818 training groups, drawn
from 9909 clean frames belonging to 24 people. Eight images per step are two
underlying frames with four light variants each, not eight independent people.
Training validation uses 48 selected groups from three validation identities.

| Branch | Validation total, step2000 -> step9909 | Output MAE | Base MAE |
| --- | --- | --- | --- |
| frozen | .08731 -> .10048 | .02950 -> .03143 | .04400 -> .04699 |
| encoder | .08667 -> .10166 | .02964 -> .03141 | .04425 -> .04745 |

Training improves while this validation subset worsens. Increasing steps alone
is not supported. This is consistent with overfitting/distribution mismatch,
but does not uniquely identify a causal component.

`oracle_adv` compares with a global scalar fitted using the ground-truth target.
It is a privileged diagnostic, not a deployed automatic-exposure algorithm.
Negative advantage does not prove that an algorithm using only input/reference
images would outperform the model. Nonetheless, weak local-light visual recovery
is real and must be addressed.

A single-channel SPATIAL gain map can brighten one cheek and darken the other.
Only a spatially constant gain is global exposure. Prior explanations conflating
single-channel gain with exposure-only behavior were incorrect. Exact input
chromaticity preservation, however, cannot correct colored incident illumination
or recover channels whose information has already clipped.

## RRNet evidence and limits

RRNet predicts enhancement lighting parameters; its parameters are not guaranteed
to be a unique physical estimate of the photographed illumination. Reference
transfer requires an additional, calibrated interpretation. The project's later
relative-light supervision partly addresses this distinction, but the paper's
success alone does not establish arbitrary cross-person reference matching.

The paper's AGM is A=I-Z, trained within its enhancement framework; it does not
specify this project's separate clean-base AGM training. Without direct intrinsic
ground truth, the inferred intermediate A need not be a physical albedo.

The paper's reported AGM ablation improves NIQE from 4.07 to 3.71 and FID from
23.24 to 23.05. It is evidence for that enhancement benchmark, not evidence that
AGM guarantees reference-light transfer or temporal stability. Missing module,
data and training details constrain exact reproduction.

Source: https://arxiv.org/html/2601.01865v1

## Bounded next experiment

1. Resolve geometry coordinates, RGB information access and compositing consistency.
   Verify normal resolution invariance and target construction numerically before
   spending time on learning. Keep a fresh, fixed synthetic diagnostic set.
2. On exactly matched examples compare: unchanged input; deployable input/reference
   exposure baseline; target-informed scalar oracle; target-informed virtual-light
   fit; and a low-resolution spatial-gain fit. Oracles diagnose representation
   limits and must not be presented as deployable methods.
3. Retain the current additive AGM decoder and test a small shallow RGB feature
   branch, without simultaneously adding another exposure head, GAN or several
   new objectives. Use 8-16 fixed base frames with clearly different spatial lights
   and a declared small-run budget (e.g. at most 2000 steps). Compare with the
   unchanged decoder under the same corrected synthetic data. First establish
   training-set fitting, then held-out identities. Do not interpret a training-set
   fit as generalization.
4. Gate full training on visible correction of local shadows and preservation of
   eyes, lips and hair, plus same-person multi-input agreement, target fidelity and
   held-out improvement. Provisional unsaturated foreground base MAE < .01 is a
   diagnostic target for the tiny training set, not a universal quality threshold.
5. If virtual-light fitting itself fails while the spatial map can fit, change the
   illumination representation. If representation fits but the neural predictor
   fails even on the tiny set, debug conditioning/optimization/decoder. If only
   held-out people fail, prioritize data diversity and pretrained priors. If
   known target parameters work but reference-image targets fail, work on reference
   estimation rather than another AGM redesign.

Do not run more full-epoch A/B jobs unchanged. There should be at most this bounded
repair/verification stage before choosing a new primary architecture.

## Other references worth comparing

- DPR, ICCV 2019: official inference code, model files and data-generation code.
  A practical first independent baseline for lighting-conditioned portrait editing.
  Its official demo predicts Lab luminance and retains input a/b, demonstrating
  that one output channel does not imply global exposure. Full deployment speed,
  video stability and training-code completeness still require checking.
  https://github.com/zhhoper/DPR
  https://raw.githubusercontent.com/zhhoper/DPR/master/testNetwork_demo_512.py
- HDRNet, SIGGRAPH 2017: use its low-resolution prediction/high-resolution local
  transform design as an efficiency reference. It is not a ready-made reference
  portrait relighting model, and the official codebase is old/archived.
  https://github.com/google/hdrnet
- Real-time 3D-aware Portrait Video Relighting, CVPR 2024: official code and
  pretrained model links, reported 32.98 fps under its conditions. It reconstructs
  portraits with a NeRF representation; identity/detail and preprocessing costs
  must be assessed before deployment. It is an informative quality/control
  comparison, not a drop-in 1080p speed promise.
  https://github.com/GhostCai/PortraitRelighting

Reference choice: use RRNet as one efficiency/rendering reference rather than the
sole architectural commitment; prioritize an official DPR inference comparison.
Do not use unverified generated faces as exact pixel-level ground truth.
