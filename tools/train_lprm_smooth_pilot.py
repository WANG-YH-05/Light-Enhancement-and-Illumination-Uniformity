"""Cached train-only LPRM diagnostic; evaluate separate validation identities.

Encoder-only warm start, reset heads, keep freshly fitted decoder statistics.
No AGM, image residual, gain head or deployment temporal changes.
"""
import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
import time

import torch
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rrnet.config import load_config, model_kwargs
from rrnet.losses import luminance
from rrnet.reference_relative_model import ReferenceRelativeRRNet
from tools.fit_grouped_light_statistics import dataset


def objective(model, group, config):
    prediction = model.base.lprm(group['input'])
    illumination, _ = model.illumination_from_theta(group['depth'], prediction['theta'])
    light = luminance(illumination)
    truth, mask = group['light'], group['mask']
    def mean(x):
        return (x * mask).sum((2, 3), keepdim=True) / mask.sum((2, 3), keepdim=True).clamp_min(1)
    def masked(x):
        return (x * mask).sum() / mask.sum().clamp_min(1)
    pred_mean, true_mean = mean(light).clamp_min(1e-4), mean(truth).clamp_min(1e-4)
    shape, true_shape = light / pred_mean, truth / true_mean
    exposure = (pred_mean.log() - true_mean.log()).abs().mean()
    shape_loss = masked((shape - true_shape).abs())
    gradient = light.new_zeros(())
    for scale in (1, 2, 4):
        a = F.avg_pool2d(shape, scale)
        b = F.avg_pool2d(true_shape, scale)
        m = F.avg_pool2d(mask, scale)
        for axis in (-1, -2):
            error = (torch.diff(a, dim=axis) - torch.diff(b, dim=axis)).abs()
            weight = m[..., 1:] * m[..., :-1] if axis == -1 else m[..., 1:, :] * m[..., :-1, :]
            gradient = gradient + (error * weight).sum() / weight.sum().clamp_min(1)
    theta = F.smooth_l1_loss(prediction['theta_normalized'],
                            (group['theta'] - model.base.lprm.denormalize.mean)
                            / model.base.lprm.denormalize.std)
    total = (config['lambda_exposure'] * exposure + config['lambda_shape'] * shape_loss
             + config['lambda_gradient'] * gradient + config['lambda_theta'] * theta)
    ambient = luminance(prediction['theta'][:, -3:, None, None])
    return dict(total=total, exposure=exposure, shape=shape_loss, gradient=gradient,
                theta=theta, ambient_share=masked(ambient / light.clamp_min(1e-4)))


def cache(model, config, split, count, device):
    ds = dataset(config, split)
    groups, manifest = [], []
    with torch.no_grad():
        for person in sorted(set(ds.people)):
            index = ds.people.index(person) * ds.variants_per_source
            for variant in range(count):
                item = ds[index + variant]
                uniform = bool(item['uniform_group'][0])
                parts = []
                for prefix, n in (('source', 4 if uniform else 1),
                                  ('reference', 1 if uniform else 4)):
                    clean = item[prefix + '_clean'][:n].to(device)
                    mask = item['source_light_mask' if prefix == 'source' else 'reference_mask'][:n].to(device)
                    person_mask = item['relight_mask' if prefix == 'source' else 'reference_relight_mask'][:n].to(device)
                    theta = item[prefix + '_theta_target'][:n].to(device)
                    depth = model.depth(clean)
                    light, _ = model.illumination_from_theta(depth, theta)
                    image = model.renderer.blend_relight(
                        clean * light, clean, person_mask).clamp(0, .999)
                    parts.append(dict(input=model.prepare_light_input(image, mask), depth=depth,
                                      theta=theta, light=luminance(light), mask=mask))
                # Store CPU caches; only one five-image group occupies GPU per step.
                groups.append({k: torch.cat([p[k] for p in parts]).cpu() for k in parts[0]})
                manifest.append(dict(split=split, source_person=person,
                                     reference_person=item['reference_person_id'], index=index + variant,
                                     uniform=uniform))
            print(f'Cached {split} {person}', flush=True)
    return groups, manifest


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--encoder-from', required=True)
    p.add_argument('--parameterization', choices=('smooth_physical', 'affine'))
    p.add_argument('--output-root')
    p.add_argument('--steps', type=int)
    p.add_argument('--log-every', type=int)
    p.add_argument('--validate-every', type=int)
    a = p.parse_args()
    c = load_config(a.config)
    d = c['diagnostic']
    for argument, key in ((a.steps, 'iterations'), (a.log_every, 'log_every'),
                          (a.validate_every, 'validate_every')):
        if argument is not None:
            if argument < 1:
                p.error(f'{key} must be positive')
            d[key] = argument
    if a.parameterization:
        c['model']['light_parameterization'] = a.parameterization
    if a.output_root:
        d['output_dir'] = a.output_root
    torch.manual_seed(20261001)
    torch.set_num_threads(4)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ReferenceRelativeRRNet(**model_kwargs(c, a.config)).to(device).eval()
    checkpoint = torch.load(a.encoder_from, map_location='cpu', weights_only=False)
    state = checkpoint.get('model', checkpoint)
    prefix = 'base.lprm.encoder.'
    encoder = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
    model.base.lprm.encoder.load_state_dict(encoder, strict=True)
    # Do not import old decoder buffers or old heads under new output semantics.
    for layer in (model.base.lprm.r0[1], model.base.lprm.r1):
        nn.init.normal_(layer.weight, std=.001)
        nn.init.zeros_(layer.bias)
    output = Path(d['output_dir']) / datetime.now().strftime('run_%Y%m%d_%H%M%S_%f')
    output.mkdir(parents=True, exist_ok=False)
    def log(message):
        line = f'[{datetime.now():%Y-%m-%d %H:%M:%S}] {message}'
        print(line, flush=True)
        with (output / 'train.log').open('a', encoding='utf-8') as f:
            f.write(line + '\n')
    log(f'Output: {output.resolve()} | decoder={c["model"]["light_parameterization"]} '
        '| encoder warm start; heads reset; FP32; fixed BN')
    train, train_manifest = cache(model, c, 'train', d['train_groups_per_person'], device)
    val, val_manifest = cache(model, c, 'val', d['validation_groups_per_person'], device)
    if {r['source_person'] for r in train_manifest} & {r['source_person'] for r in val_manifest}:
        raise ValueError('Training/validation identity overlap')
    (output / 'experiment.json').write_text(json.dumps(dict(
        config=c, encoder_from=str(Path(a.encoder_from).resolve()),
        image_pipeline='train.py soft-mask blend; clamp [0, 0.999]',
        manifest=train_manifest + val_manifest,
        scope='Small cached lighting-estimator diagnostic, not formal training or untouched test'), indent=2), encoding='utf-8')
    def move(g):
        return {k: v.to(device) for k, v in g.items()}
    def evaluate(step):
        model.eval()
        means_by_split = {}
        rows_by_split = {}
        with torch.no_grad():
            for split, groups in (('training', train), ('validation', val)):
                rows = [{k: float(v) for k, v in objective(model, move(g), d).items()} for g in groups]
                means_by_split[split] = {k: sum(r[k] for r in rows) / len(rows) for k in rows[0]}
                rows_by_split[split] = rows
        with (output / 'metrics.jsonl').open('a', encoding='utf-8') as f:
            f.write(json.dumps(dict(step=step, **means_by_split, rows=rows_by_split['validation'])) + '\n')
        for split, means in means_by_split.items():
            log(split.upper() + ' ' + str(step) + ' ' + ' '.join(f'{k}={v:.5f}' for k, v in means.items()))
    evaluate(0)
    optimizer = torch.optim.Adam(model.base.lprm.parameters(), lr=d['learning_rate'])
    start = time.monotonic()
    generator = torch.Generator().manual_seed(20261001)
    order = []
    for step in range(1, d['iterations'] + 1):
        if not order:
            order = torch.randperm(len(train), generator=generator).tolist()
        model.train()
        for module in model.base.lprm.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()
        optimizer.zero_grad(set_to_none=True)
        metrics = objective(model, move(train[order.pop()]), d)
        if not all(torch.isfinite(v) for v in metrics.values()):
            raise RuntimeError('Non-finite diagnostic loss')
        metrics['total'].backward()
        torch.nn.utils.clip_grad_norm_(model.base.lprm.parameters(), 5.)
        optimizer.step()
        if step % d['log_every'] == 0:
            elapsed = time.monotonic() - start
            log(f'TRAIN {step}/{d["iterations"]} total={float(metrics["total"]):.5f} '
                f'elapsed={elapsed:.0f}s ETA={elapsed / step * (d["iterations"]-step):.0f}s')
        if step % d['validate_every'] == 0 or step == d['iterations']:
            evaluate(step)
            torch.save(dict(model=model.state_dict(), config=c, step=step, diagnostic_only=True,
                            optimizer=optimizer.state_dict(), sampling_generator=generator.get_state(),
                            sampling_order=list(order), torch_rng=torch.get_rng_state(),
                            cuda_rng=torch.cuda.get_rng_state_all() if device.type == 'cuda' else None),
                       output / f'lprm_diagnostic_step_{step:06d}.pt')
    log('Diagnostic complete; no deployment weights replaced')


if __name__ == '__main__':
    main()
