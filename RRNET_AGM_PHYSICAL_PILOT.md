# RRNet-style AGM known-light pilot

This is a separate experiment. It does not modify the existing reference-relative
model or its checkpoints. `rrnet/agm.py` is a reconstruction of the paper's
three-channel AGM, **not official RRNet code**.

## What is trained

The pretrained RepViT/LPRM encoder and Depth Anything V2 are frozen. Only the
three-channel AGM head is updated. The dedicated AGM loader reads only one
MEAD person's clean frame and its masks; it does not read a different person's
reference image. For each clean frame it chooses several different synthetic
source lights and one shared, randomly sampled target light.
The depth-aware renderer creates the input and target on the GPU using the
known light parameters. One member of each group is an unchanged-input anchor.
The path optimized is:

`synthetically lit input -> AGM base -> renderer(known target theta) -> target`.

The clean MEAD frame is used as the base **only for this constructed synthetic
pair**; it is not a measured intrinsic albedo map. Consequently, success here
would establish that AGM can invert these added virtual lights, not that it can
already remove arbitrary real-world illumination. Estimated reference theta is
deliberately excluded so its error cannot hide an AGM failure.

## Run in PowerShell from the Model folder

```powershell
Set-Location "E:\Lighting Enhancement Project\RRNet\Model"
& "E:\conda_envs\iclight\python.exe" train_agm_physical.py --config "configs\rrnet_agm_physical_pilot.yaml"
```

By default this runs 2,000 steps. Each run gets a timestamped directory under
`outputs/rrnet_agm_physical_pilot/` with `train.log`, metric JSONL files and
`agm_step_0000500.pt` etc. Checkpoints contain only the AGM head plus training
state; the frozen encoder checkpoint path is recorded in the checkpoint and
`run_config.json`. Do not pass these head-only checkpoints to the current video
inference script. This pilot has **no reference-video inference integration**.

For a quick execution check without a full run:

```powershell
& "E:\conda_envs\iclight\python.exe" train_agm_physical.py --config "configs\rrnet_agm_physical_pilot.yaml" --steps 1 --validation-batches 1 --output-dir "outputs\rrnet_agm_physical_smoke"
```

## How to judge the pilot

- `out` is masked RGB error after AGM plus virtual-light rendering.
- `base` measures recovery of the known synthetic base frame.
- `spatial` penalizes a wrong spatial correction pattern; a single scalar cannot
  generally remove side light.
- `chroma` checks skin chromaticity in unclipped regions.
- `identity` measures damage to the unchanged-input anchor.
- `oracle_adv` compares output against a **target-informed** global scalar;
  that scalar is not available during deployment, and this diagnostic alone
  must not select the model.

Use validation (held-out MEAD people) to select a checkpoint. Before treating
the experiment as successful, inspect side-light and normal-light images,
including colour artifacts. Do not select on the previously inspected test
split. If the spatial result is not visibly better than exposure adjustment,
stop rather than extending the run.

## Controlled encoder A/B continuation

Both branches initialize the AGM from the completed 2,000-step pilot and the
encoder from the same existing reference-relative model. Both use the new
source-only loader, identical RNG seed, data order, loss, and one complete
pass through the training groups (9,909 optimizer steps at batch size 2).
Each batch contains two groups, i.e. eight images because each group has four
images. The sole experimental difference is whether encoder weights receive
gradients; its BatchNorm running statistics stay frozen in both branches.
Run the two commands **sequentially** on one GPU:

```powershell
& "E:\conda_envs\iclight\python.exe" train_agm_physical.py --config "configs\rrnet_agm_ab_frozen.yaml"
& "E:\conda_envs\iclight\python.exe" train_agm_physical.py --config "configs\rrnet_agm_ab_encoder.yaml"
```

Outputs are separate timestamped runs under `outputs/rrnet_agm_ab_frozen/`
and `outputs/rrnet_agm_ab_encoder/`. The new step numbers count **additional**
steps from the pilot checkpoint; `agm_step_0009909.pt` is not the old pilot's
9,909-step checkpoint. Validation and checkpoints are every 2,000 steps, and
console summaries are every 500.
The trainable-encoder checkpoint includes its updated encoder state. To
inspect either branch after training, use `tools/inspect_agm_physical.py`
with its run directory and `--steps 2000 4000 6000 8000 9909`. Compare the branches on
identical held-out people, especially normal-light recovery and identity
damage. The target-informed scalar and known-light inverse are diagnostic
oracles, not real-time deployment baselines.
