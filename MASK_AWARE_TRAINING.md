# Mask-aware MEAD V2 training

Use `configs/rrnet_mead_v2_weighted.yaml` for the regenerated dataset. Do not
start this run until dataset generation has completed, `metadata.csv` exists,
and the V2 lighting parameter statistics have been fitted.

## Data path and sampling

The configuration reads:

```text
E:/Lighting Enhancement Project/RRNet/MEAD for RRNet/dataset/metadata.csv
```

Training uses weighted replacement sampling: identity is 30%, and each of the
six degradation categories is approximately 11.67%. Validation is deterministic
and unweighted.

## Mask semantics

- `relight_mask` is the soft foreground alpha used for output compositing and
  foreground reconstruction loss.
- `skin_mask` adds a stronger reconstruction constraint to visible skin.
- The displayed output is clamped to `[0, 1]`, but losses use the pre-clamp
  foreground rendering so saturated predictions retain a corrective gradient.
- Pixels where `relight_mask == 0` are copied exactly from the degraded input.

The mask-aware objective is:

```text
total = relight
      + lambda_skin * skin
      + roi
      + lambda_highlight * highlight
      + lambda_background * background
      + lambda_reg * regularization
```

The highlight term is target-aware: it penalizes output luminance only when it
exceeds both the configured threshold and the target luminance plus a margin.

## Logged metrics

`train.log`, `train_metrics.jsonl`, and `val_metrics.jsonl` record:

- `total`: weighted objective used for optimization
- `relight`: reconstruction error inside the person mask
- `skin`: reconstruction error inside the skin mask
- `highlight`: target-aware excessive-highlight penalty
- `background`: difference from input outside/around the soft relight boundary
- `roi`: paper-style depth/luminance weighted reconstruction term
- `regularization`: lighting-parameter constraint

## Launch after generation

Calibrate equation (2) first:

```powershell
cd "E:\Lighting Enhancement Project\RRNet\Model"
& "E:\conda_envs\iclight\python.exe" calibrate_lighting_stats.py `
  --config "configs\rrnet_mead_v2_weighted.yaml" `
  --samples 1000 --batch-size 8 --optimization-steps 100 `
  --output "checkpoints\mead_v2_masked_theta_stats_1000.npz"
```

Then launch training:

```powershell
cd "E:\Lighting Enhancement Project\RRNet\Model"
& "E:\conda_envs\iclight\python.exe" train.py --config "configs\rrnet_mead_v2_weighted.yaml"
```

Run tests first:

```powershell
& "E:\conda_envs\iclight\python.exe" -m pytest -q
```
