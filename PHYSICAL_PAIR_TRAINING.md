# Physical-pair training for reference lighting

This is an isolated experimental path for fixing the observed low-frequency,
partly non-physical virtual-light decomposition. It does not replace or modify
the behavior of existing configs and checkpoints.

## Why this path exists

Diagnostics on the 14k reference-relative checkpoint found that:

- many dark/backlit samples were dominated by the ambient term;
- some predicted virtual lights acted as negative compensation;
- effective gain maps were spatially varying but mostly broad, smooth gradients;
- local facial illumination around the eyes, nose, forehead, and jaw was not
  represented well enough for convincing reference-light uniformity.

The old training pairs supervise rendered images and inferred illumination maps,
but the underlying light decomposition is not unique. A model can lower those
losses without recovering a stable set of virtual lights.

## Exact physical pair

For two different clean MEAD participants A and B, training samples are rendered
on GPU as:

```text
input     = clean_A * render(depth_A, theta_in)
reference = clean_B * render(depth_B, theta_ref)
target    = clean_A * render(depth_A, theta_ref)
```

The same person mask blending used by inference keeps the background unchanged.
`theta_in` and `theta_ref` are retained as exact labels. Light slots are anchored
to a 3x3 image-space grid so direct theta supervision has no permutation
ambiguity.

`theta_in` deliberately includes normal, dark, over-bright, side, top, back,
and mixed illumination. A reference participant is never assigned a
side/back/top/mixed light. The sampler supports **dark**, **normal**, and
**bright** reference regimes, but the current pilot configuration deliberately
enables only **dark** and **normal** (`bright: 0.00`); this avoids treating an
overexposed image as a desirable meeting master light. Their probabilities are
configured by `data.reference_lighting_weights`.

## Anti-shortcut grouping

The physical pilot uses four-member groups. In the main 80% of groups, one
source portrait receives four deliberately distinct input lights while one
different-person reference portrait and one `theta_ref` stay fixed. All four
members therefore have the same exact target light. `Lcons` explicitly penalizes
their output-light disagreement.

The remaining 20% are reference-sensitivity groups: the source input stays
fixed while dark and normal reference lights alternate. Their targets change
with the reference and they are excluded from `Lcons`. This prevents the easy
but incorrect solution of ignoring the reference image. The two group types are
kept separate by `uniform_group`, so their objectives never conflict.

## Isolation from the historical model

- Existing configs leave `enforce_physical_parameters` disabled by default.
- The new renderer projection is enabled only by the physical configs.
- Existing checkpoints are never overwritten.
- The new task has its own dataset class and output directory.
- AGM remains disabled.
- RGB channels still share the same spatial transfer gain.

When enabled, the renderer projects colors, attenuation, and ambient light to
non-negative values, normalizes directions, and bounds positions. This prevents
negative virtual lights from cancelling an oversized ambient term.

## Stage 1: physical pilot

The recommended initialization keeps the old lighting encoder but deliberately
does not load the old regression heads or fitted parameter statistics. The new
heads first train alone for 300 steps, after which the encoder is unfrozen.

```powershell
Set-Location "E:\Lighting Enhancement Project\RRNet\Model"

& "E:\conda_envs\iclight\python.exe" train.py `
  --config "configs\rrnet_mead_reference_relative_physical_pilot.yaml" `
  --init-from "outputs\rrnet_mead_reference_relative_darklift_scratch\run_20260911_185954\rrnet_step_0014000.pt"
```

This is a 3000-step pilot. Its outputs are stored under:

```text
outputs/rrnet_mead_reference_relative_physical_pilot/run_<timestamp>/
```

Do not proceed based only on total loss. Inspect reference sensitivity, gain-map
spatial structure, skin-color stability, overexposure, and video flicker.

## Stage 2: MEAD degradation adaptation

Only after the pilot passes visual inspection, initialize the second stage from
the selected pilot checkpoint:

```powershell
& "E:\conda_envs\iclight\python.exe" train.py `
  --config "configs\rrnet_mead_reference_relative_physical_finetune.yaml" `
  --init-from "outputs\rrnet_mead_reference_relative_physical_pilot\run_<timestamp>\rrnet_step_0003000.pt"
```

Stage 2 adapts the physically initialized model to the existing MEAD degradation
pipeline at a lower learning rate. It does not have exact theta labels, so the
direct theta loss weights are zero in that config while the physical renderer
remains enabled.

For stage 2, the reference side is currently restricted to `underexposed_cool`
(dark) and `identity` (normal); all other generated categories, including
`warm_overexposure`, remain source-only. Since this architecture uses achromatic
illumination and preserves source chromaticity, these labels act as brightness
regimes rather than a request to transfer a warm/cool skin tone.

## New logged values

- `Ltheta source/reference`: direct normalized theta regression error, available
  in stage 1;
- `Lin`, `Lref`, `Ltgt`: source, reference, and target illumination-map errors;
- `Lgain`: log transfer-gain error;
- `Lgrad`: multiscale gain-gradient error.

The new training path has completed a real CUDA forward, loss, and backward
smoke test with finite gradients. Unit tests also cover valid theta sampling and
the prevention of negative-light cancellation.

## 2026-10-01 diagnostic audit (not a new recommended model)

The limited physical-validation loop previously consumed the first 30 sorted
source frames, all from `MEAD_front_11`. Physical validation now deterministically
balances the requested sample budget across identities and frame positions.
For the current 30-group configuration it evaluates 10 groups each from persons
11, 14, and 37. Historical validation values are therefore not directly
comparable with the corrected validation values. This fix does not create an
untouched final test set or undo earlier test-set exposure.

### Controlled BatchNorm experiment

Both diagnostic runs start from the same shape-stage2 8000-step checkpoint,
use seed 20261001, the same samples, learning rate, losses, and 200-step schedule.
Only `training.freeze_lprm_bn` differs. This flag fixes running statistics while
leaving BatchNorm affine weights and the other parameters trainable; it does not
freeze the whole encoder. It defaults to false in existing configurations.

| Checkpoint | Balanced val total | Relight L1 | Log gain error | Shape consistency |
|---|---:|---:|---:|---:|
| Original 8000 steps | 1.079204 | 0.041014 | 0.217970 | 0.036073 |
| Fixed BN, +200 steps | 0.988729 | 0.046936 | 0.231174 | 0.036338 |
| Ordinary BN, +200 steps | 1.315439 | 0.047578 | 0.285023 | 0.038028 |

Fixing statistics avoids the deterioration seen in this ordinary-BN continuation,
but does not improve reconstruction over the original checkpoint: relight L1
increases approximately 14.4%, even though total decreases approximately 8.4%.
Do not promote this checkpoint or launch a long fixed-BN run based on total alone.
The fixed run changes 387 non-running-stat LPRM tensors and none of its 321 BN
running-stat buffers. This confirms that training updates occurred with fixed
statistics. All 321 buffers change in the ordinary-BN control.

The controlled visual grid has 12 rows: persons 11, 14, and 37, four input lights
each. Its columns are reference, input, synthetic target, original 8000-step
output, fixed-BN continuation, and ordinary-BN continuation. The fixed-BN version
still fails to bring every input to its shared target; the darker inputs remain
visibly darker. These images are diagnostics, not final generalization claims.

Artifacts (relative to this Model directory):

- `outputs/rrnet_mead_reference_relative_shape_stage2_8k/run_20260929_152638/diagnosis/balanced_val.json`
- `outputs/rrnet_bn_diagnostic_fixed_200/run_20261001_134809/val_metrics.jsonl`
- `outputs/rrnet_bn_diagnostic_baseline_200/run_20261001_134952/val_metrics.jsonl`
- `outputs/rrnet_bn_diagnostic_fixed_200/run_20261001_134809/inspection/three_identity_comparison.jpg`

### What the renderer diagnosis establishes

On six deterministic train/val groups, supplying known synthetic theta values
instead of predicted values produces face-region MAE approximately 0.00003 to
0.00121. Therefore this transfer renderer can reproduce these particular
synthetic targets; this does not establish real-world intrinsic-light recovery.
An optimally selected scalar exposure also beats the predicted model on these
groups. That scalar uses the correct target to select its gain, so it is an
oracle diagnostic baseline, not a deployable comparison.

Batch-statistics inference substantially changes predictions, but helps some
samples and harms others. Thus BN sensitivity is demonstrated, not proved to
be the only cause. Separate coarse/full-resolution running statistics plus
60 training-only calibration batches also failed to improve the small val
diagnostic on average. `model.split_resolution_bn` is experimental, defaults
to false, and is not enabled in the recommended existing configurations.

AMP now uses the current `torch.amp` API. The learning-rate scheduler advances
only when GradScaler has not skipped the optimizer update. No inference temporal
smoothing, AGM activation, gain limits, or existing checkpoints were changed.
The validation/BN and existing RRNet unit suites pass: 50 tests.

Next experiments should target predicted illumination-map shape/exposure errors
and be accepted only if both the balanced visual grid and component metrics
improve. Do not infer success from total loss alone, blindly extend training,
or add another image-residual branch based on these results.

## Fixed-group memorization diagnostic, 2026-10-01

Executed `tools/overfit_physical_groups.py` for 600 updates from the unchanged
shape-stage2 8000-step checkpoint. No deployment model/config was modified.
This deliberately narrow experiment uses train persons 0, 1, and 10 as inputs,
and train person 12 as the single different-person reference. Every input
person has four fixed source-light variants, paired with both a fixed normal
and a fixed dark reference light: six groups, 24 input/reference pairs. The
normal/dark groups use identical source variants. Within a reference mode,
all three source people share the same reference frame and theta. Source,
reference, exact synthetic targets and frozen depth are cached. FP32, fixed
BN statistics, Adam at 1e-4, existing full loss, all LPRM parameters trainable.
This is not an epoch-based generalization experiment or a production weight.

| Six training groups, averaged | Before | After 600 |
|---|---:|---:|
| Face-region RGB MAE | 0.032314 | 0.009667 |
| Normalized illumination-shape loss | 0.047438 | 0.018110 |
| Log gain error | 0.185078 | 0.049546 |
| Within-group face-luminance span | 0.025829 | 0.013361 |
| Output shape consistency loss | 0.052869 | 0.023157 |

At step 500 the face MAE was lower, 0.00561, than at the final step. Final
results are therefore not reported as a best-checkpoint claim. Learning-rate
1e-4 with a small cyclic sample set still causes fluctuations. Visual inspection
shows substantial improvement and reference responsiveness, but person 0 is
still systematically too bright and has residual variation across its inputs.
Predicted spatial illumination maps are not constant, yet do not perfectly
match known maps. The results establish partial small-set fitting capacity,
not successful real-world lighting unification.

On the unchanged balanced 30-group validation, relight L1 increases from
0.041014 to 0.076704; total increases from 1.079204 to 1.277707. This is expected
to be a risk of deliberately fitting three people and one reference, and
cannot by itself identify the cause of the larger model's generalization error.
Do not replace the existing weight with this diagnostic checkpoint.

Bounds audit: normal-reference groups have no face pixels whose known gain
exceeds the 0.4..5.0 range. Dark-reference groups have approximately 19.2%,
8.0%, and 1.4% face-mask-weighted pixels below the 0.4 minimum for persons 0,
1, and 10 respectively. No group requires gain above 5. Known-theta oracle
face MAE is approximately 0.00006..0.00203, still much below several predicted
outputs. Thus bounds affect some dark targets, but increasing maximum gain is
not the solution to these particular errors.

Artifacts are under
`outputs/rrnet_fixed_group_diagnostic/run_20261001_144025_781687/`:
`experiment.json` records exact training frames/identities/theta;
`metrics.jsonl`, `summary.json`, and `train.log` record the experiment;
`balanced_val.json` records unchanged validation; `gain_bounds_audit.json`
records bounds and per-input errors. Six `*_comparison.jpg` grids have columns:
reference, input, synthetic target, before, after, known-theta oracle, source
shape target, source shape prediction. Shape grayscale uses 0.75..1.25 and
is contrast-expanded for diagnosis; it is not the raw RGB image.

The appropriate next step is a controlled supervision/data-generalization
experiment with the unchanged inference structure, rather than assuming
another dense residual/AGM branch is necessary. In particular, compare direct
theta supervision weights while keeping illumination-map/gain-map and
same-reference group supervision: many virtual-light parameters can explain
similar images, so parameter-label error and output quality need separate
inspection. This is a proposed experiment, not a confirmed cause or a fix.

## Theta-supervision weight ablation protocol, 2026-10-01

Two separately saved 1000-step runs start from the identical stage2 8000-step
weight, not the memorization experiment. Configurations are
`configs/rrnet_theta_supervision_control_1000.yaml` (source/reference theta
weights 0.5/0.5) and `configs/rrnet_theta_supervision_weak_1000.yaml` (0.1/0.1).
Their parsed configurations were verified to be identical except these two
weights and output directory. Both use seed 20261001, fixed BN statistics,
learning rate 1.5e-5, 100-step warmup/cosine schedule, group size 4, 20%
reference-sensitivity groups, and unchanged image/map/gain/consistency losses.
Validation remains the same balanced 30 groups; test is not used.

Raw training/validation totals must not be compared directly across weights.
`tools/compare_theta_supervision_runs.py` computes the weak model's common total
by adding `0.4 * (theta_source + theta_reference)` to its recorded total,
thereby evaluating both under theta weights 0.5/0.5. Component losses and visual
comparisons remain the primary criteria. This single-seed short experiment is
not evidence of statistical significance and cannot establish final test quality.

### Completed ablation results

| Weight / additional steps | Common total | Relight L1 | Log gain | Illumination shape | Group consistency |
|---|---:|---:|---:|---:|---:|
| Initial 8000-step checkpoint | 1.079204 | 0.041014 | 0.217970 | 0.043516 | 0.007948 |
| Control 0.5 / +500 | 0.930906 | 0.046026 | 0.232947 | 0.044644 | 0.007090 |
| Weak 0.1 / +500 | 0.927818 | 0.040891 | 0.208182 | 0.044452 | 0.006978 |
| Control 0.5 / +1000 | 0.913438 | 0.045196 | 0.228844 | 0.044347 | 0.006824 |
| Weak 0.1 / +1000 | 0.973414 | 0.045967 | 0.232985 | 0.043832 | 0.007049 |

Weak-theta +500 is only 0.3% better than the original in relight L1, although
it beats the same-step control and improves gain-log error. That improvement
does not persist at +1000. Illumination-shape error does not beat the original
at either point. Visual inspection of the two same-step grids finds no clear,
consistent improvement: some dark inputs remain too dark and distinct source
lights do not reliably reach their shared target. Both continuations improve
some consistency metrics but neither demonstrates successful unification.
Keep the original checkpoint; do not adopt theta weights 0.1 as a confirmed fix.

All 321 LPRM BN running-stat buffers remain unchanged in both runs. The final
AMP scaler value is 8192 for both. Existing tests pass (50). No renderer,
depth size, gain limits, inference smoothing, masks, or architecture changed.

Run directories:

- `outputs/rrnet_theta_supervision_control_1000/run_20261001_184140/`
- `outputs/rrnet_theta_supervision_weak_1000/run_20261001_185414/`

Shared report: `outputs/rrnet_theta_supervision_ablation/comparison.json`.
It includes all four checkpoints per run and the best-by-validation-L1
checkpoints (control +750; weak +500). They are diagnostic selection, not
untouched test results. Same-step grids are `comparison_step_0000500.jpg`
and `comparison_step_0001000.jpg` in that folder. Columns: fixed reference,
input, synthetic target, original 8000-step model, theta 0.5 continuation,
theta 0.1 continuation. Each has three validation identities and four source
lights per identity.

This short ablation does not establish excessive direct theta supervision
as the main cause. Further work should isolate cross-identity illumination
estimation from objective weighting before another full training run.

## Fixed-theta cross-identity audit, 2026-10-01

Read-only evaluation of the original shape-stage2 8000-step weight used
six train identities (0, 13, 19, 27, 35, 9) and all three val identities
(11, 14, 37), two distinct frames per identity. No training occurred and no
test identities were used. Every frame received the identical four theta
vectors: dark reference, normal reference, and two directional source lights.
The source lights were selected for higher spatial variation among 12
deterministic samples from the current physical sampler; they are stress cases,
not an estimate of average dataset performance.

Predicted and true theta are rendered on the *same* anchor depth/face mask
before comparing exposure and normalized shape. This prevents legitimate
geometry differences between people from explaining differences in that
comparison. Same theta refers to a known synthetic multiplier over each
MEAD clean frame, NOT measured identical intrinsic scene illumination.

Validation normal-reference common exposure should always be 0.887607;
predictions range 0.529153..0.893750 across six frames, with mean relative
error 24.2% and coefficient of variation 18.0%. Clean-face luminance and
predicted common exposure have descriptive correlation 0.916. This is
consistent with identity/base-image brightness confounding, but is not causal
proof that skin tone alone is responsible (three identities, repeated frames,
pose/material/studio-light differences remain).

The directional cases also show strongly flattened predicted light maps:

| Light, val on common geometry | True normalized spatial std | Predicted mean std |
|---|---:|---:|
| Source A | 0.180716 | 0.042804 |
| Source B | 0.141829 | 0.040558 |

Thus predicted variation is approximately 24% and 29% of the known variation.
Train identities show flattening as well (approximately 12% and 15%). This
means the observation is not exclusively a held-out-identity phenomenon.
The current renderer can express the target maps, but the predicted parameters
fail to reproduce them in these particular directional stress cases.

For a separate intervention, hold one train-person-0 input and its depth fixed;
switch normal/dark reference images across all audited identities. Normalized
shape is `light / masked mean(light)`. Correct exposure replaces predicted
source/reference means by known means while retaining predicted shapes.
Correct shape does the reverse. These use ground truth and are diagnostics,
not proposed inference postprocessing.

| Val-reference intervention, same fixed source | Face-region output MAE |
|---|---:|
| Predicted source and reference | 0.056031 |
| Correct both exposure means only | 0.024392 |
| Correct both spatial shapes only | 0.046058 |
| Correct reference light completely | 0.020910 |
| Correct source light completely | 0.047553 |
| Both known light maps | 0.000082 |

Exposure-only correction reduces error 56.5%, shape-only 17.8%. Reference
prediction is important for this fixed source, but the fractions cannot be
generalized to all sources or all real videos. The two interventions are
nonlinear and their percentage improvements should not be added.

Report and grids are under
`outputs/rrnet_identity_light_audit/run_20261001_193858_832895/`:
`report.json`, `normal_reference_identity_comparison.jpg`,
`dark_reference_identity_comparison.jpg`, and
`directional_light_shape_comparison.jpg`. The earlier run without spatial-std
columns remains preserved separately.

Next training hypothesis: explicit cross-person same-light groups, supervising
estimated illumination after projection onto one shared geometry. Retain both
same-person/different-input unification and fixed-input/different-reference
contrast. Do NOT equalize RGB face brightness across different people; that
would confuse material/skin with lighting again. This changes training
constraints/data grouping, not the inference network. It is a hypothesis to
test, not a validated fix.

## 2026-10-01: opt-in smooth light decoder and matched pilot

The LPRM gradient audit found 51–69% negative RGB intensity components in six
original-checkpoint groups. These components receive zero gradient through the
hard color clamp in the rendering branch (direct theta supervision can still
update them). Predicted ambient share was 83–91%, versus 49–73% for those groups'
known synthetic parameters. The heads remained connected to the graph.

`light_parameterization: smooth_physical` is an independent experimental mode,
not a claim of exact RRNet Eq. (2) reproduction. It uses mean-centered softplus
for intensity/attenuation/ambient, sigmoid for positions, and normalized
directions. `theta_normalized` continues to mean physical `(theta-mean)/std`;
the unconstrained network output is separately exposed as `theta_latent`.
The default `affine` path and checkpoint buffer keys remain unchanged.
The original stage2 8k checkpoint loaded strictly with all keys matched;
58 regression/unit tests passed, including sampler RNG equivalence.

Calibration: `tools/fit_grouped_light_statistics.py` used 4,000 **train-only**
groups / 20,000 unique encoded source/reference instances. It reuses the exact
dataset sampler, including source-diversity rejection and sensitivity groups.
Output: `checkpoints/mead_grouped_theta_stats_smooth_20261001.npz` and its JSON
provenance. No calibration from val/test. Parameter calibration alone cannot
remove identity/lighting ambiguity in MEAD.

Pilot: `tools/train_lprm_smooth_pilot.py`, configuration
`configs/rrnet_lprm_smooth_pilot.yaml`. Both decoder modes use the same calibrated
statistics, encoder-only warm start from stage2 8k, reset heads, random seed,
48 cached training groups across 24 people, six validation groups across three
separate people, 300 FP32 steps, fixed BN and direct lighting-map losses.
The synthetic input uses the same soft-mask blend and [0, 0.999] clamp as
`train.py`. Depth is frozen/cached. No AGM or image residual is added.
This is a small estimator diagnostic, not formal training or untouched testing.

| Validation metric (lower is better except shares) | Affine control, 300 | Smooth, 300 |
|---|---:|---:|
| Log exposure error | 0.178234 | 0.168438 |
| Normalized spatial shape error | 0.052150 | 0.051808 |
| Shape gradient error | 0.041776 | 0.041237 |
| Negative color component fraction | 19.14% | 0% |
| Fully clamped-off light fraction | 12.22% | 0% |
| Predicted ambient share | 77.59% | 77.12% |

Smooth initialization shape error was 0.051687; at step 300 it is 0.051808.
Thus the constraint fix eliminates this hard-clamp failure but **does not
demonstrate improved spatial relighting/generalization**. Relative to matched
affine, exposure error improves about 5.5%, while shape improves only 0.65%.
Do not promote these diagnostic checkpoints as better deployment models.
No inference latency has been benchmarked for this new mode.

Corrected-pipeline runs (use these, not the earlier exploratory hard-edge runs):
- `outputs/rrnet_lprm_smooth_pilot/run_20261001_233026_378455/`
- `outputs/rrnet_lprm_affine_control_pilot/run_20261001_233236_345359/`
- Comparison: `outputs/rrnet_lprm_decoder_comparison/run_20261001_233428_915464/`
  contains `comparison.json` and `validation_light_shapes.jpg`.

The next unresolved hypothesis is cross-person, same-known-light supervision
on shared geometry, retaining fixed-reference multi-input uniform groups and
fixed-input changing-reference groups. It is not implemented by this standalone
decoder pilot. Never equalize different people's RGB brightness as a proxy for
equal illumination. MEAD clean portraits still carry original studio shading.
Old heads/decoder buffers must not be imported under new decoder semantics;
only the encoder is warmed in this pilot. Keep deployment models/configs intact.
