# Safe AGM stage 1

This stage isolates source-light removal from virtual-light rendering and
reference transfer.  It must pass the scalar-exposure baseline before it is
connected to the reference model.

## What changed

- `rrnet/safe_agm.py` predicts a bounded one-channel log-gain map.
- The final layer is zero initialized, so training starts from the input image.
- RGB channels share the same spatial gain; skin chromaticity cannot be changed
  by the AGM.
- `rrnet/agm_data.py` groups multiple lighting degradations of the same frame.
- `rrnet/safe_agm_loss.py` directly supervises albedo, gain-map gradients,
  group consistency, chromaticity, smoothness, and background identity.
- `train_safe_agm.py` is independent of the existing RRNet/reference runs.

The historical paper-style three-channel AGM remains in `rrnet/agm.py` and is
not overwritten.

Stage 1 optimizes luminance and spatial illumination only. Chromaticity is
reported as a diagnostic but has zero weight because a shared RGB gain cannot
correct white balance without changing skin colour. A bounded global
white-balance stage, if needed, is evaluated only after spatial relighting
beats the scalar-exposure baseline.

## Pilot training

```powershell
Set-Location "E:\Lighting Enhancement Project\RRNet\Model"

& "E:\conda_envs\iclight\python.exe" train_safe_agm.py `
  --config "configs\safe_agm_stage1.yaml" `
  --init-encoder-from "outputs\rrnet_mead_reference_relative_physical_v2\run_20260925_210230\rrnet_step_0008000.pt"
```

The `adv` value in training and validation logs is:

```text
best scalar-exposure MAE - safe AGM MAE
```

Positive is better than global exposure.  Do not proceed to reference transfer
if validation `adv` is non-positive or if side-light gain correlation remains
low.

## Category evaluation

```powershell
& "E:\conda_envs\iclight\python.exe" tools\evaluate_safe_agm.py `
  --config "configs\safe_agm_stage1.yaml" `
  --checkpoint "outputs\safe_agm_stage1\RUN\safe_agm_step_0005000.pt" `
  --dataset-root "E:\Lighting Enhancement Project\RRNet\MEAD for RRNet\dataset" `
  --split test `
  --samples-per-category 100 `
  --output "outputs\safe_agm_stage1\RUN\test_metrics.json"
```

The most important fields are `advantage` and `gain_correlation`, especially
for `warm_side_light`, `window_backlight`, and `top_light_shadow`.
