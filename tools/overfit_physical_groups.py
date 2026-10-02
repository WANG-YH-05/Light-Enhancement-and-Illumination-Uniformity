"""Fixed training-only physical groups: capacity diagnostic, not deployment training."""
from __future__ import annotations

import argparse
import copy
from datetime import datetime
import json
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image, ImageDraw
import torch
from torch import nn
from torch.utils.data import default_collate

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rrnet.config import load_config, model_kwargs
from rrnet.losses import luminance
from rrnet.physical_reference_data import MEADPhysicalReferencePairs, sample_reference_theta
from rrnet.reference_relative_model import ReferenceRelativeRRNet
from rrnet.reference_relative_loss import ReferenceRelativeLoss
from train import model_prediction


def forward(model, group):
    return model.forward_physical_grouped(
        group['source'], group['extras']['reference'], 4,
        group['extras']['uniform_group_mask'], group['mask'],
        group['extras']['reference_mask'], group['extras']['source_light_mask'],
        source_depth_override=group['depth'],
        reference_depth_override=group['reference_depth'])


def losses(criterion, group, prediction):
    return criterion(prediction, group['target'], group['source'],
                     group['mask'], group['skin'], **group['extras'])


def snapshot(model, criterion, groups):
    model.eval()
    results = []
    with torch.no_grad():
        for g in groups:
            p = forward(model, g)
            metrics = {k: float(v) for k, v in losses(criterion, g, p).items()}
            face = g['extras']['source_light_mask']
            output_y = luminance(p['loss_output'])
            means = (output_y * face).sum((1, 2, 3)) / face.sum((1, 2, 3)).clamp_min(1)
            metrics['output_luma_span'] = float(means.max() - means.min())
            metrics['face_mae'] = float(
                ((p['loss_output'] - g['target']).abs() * face).sum()
                / (3 * face.sum().clamp_min(1)))
            ratio = (p['reference_illumination_on_source']
                     / p['source_illumination'].clamp_min(model.illumination_floor))
            for name, flag in (
                    ('gain_lower_bound_fraction', ratio <= model.min_transfer_gain),
                    ('gain_upper_bound_fraction', ratio >= model.max_transfer_gain)):
                metrics[name] = float((flag.float() * face).sum()
                                      / (flag.shape[1] * face.sum().clamp_min(1)))
            results.append({'person': g['person'], 'mode': g['mode'], 'metrics': metrics})
    return results


def image(tensor, low=0.0, high=1.0):
    value = ((tensor.detach().float() - low) / (high - low)).clamp(0, 1)
    if value.shape[0] == 1:
        value = value.expand(3, -1, -1)
    pixels = (value.permute(1, 2, 0).cpu().numpy() * 255).round().astype('uint8')
    return Image.fromarray(pixels).resize((180, 180), Image.Resampling.LANCZOS)


def visualize(model, groups, output):
    model.eval()
    with torch.no_grad():
        for g in groups:
            p = forward(model, g)
            oracle = model.transfer(g['source'], g['depth'],
                                    g['extras']['source_theta_target'],
                                    g['extras']['reference_theta_target'], g['mask'])
            source_true, _ = model.illumination_from_theta(
                g['depth'], g['extras']['source_theta_target'])
            target_true, _ = model.illumination_from_theta(
                g['depth'], g['extras']['reference_theta_target'])
            face = g['extras']['source_light_mask']
            def shape(x):
                x = luminance(x)
                return x / criterion_mean(x, face)
            def criterion_mean(x, mask):
                return ((x * mask).sum((1, 2, 3), keepdim=True)
                        / mask.sum((1, 2, 3), keepdim=True).clamp_min(1)).clamp_min(1e-4)
            predicted_shape = shape(p['source_illumination'])
            true_shape = shape(source_true)
            canvas = Image.new('RGB', (1460, 4 * 205 + 65), '#171717')
            draw = ImageDraw.Draw(canvas)
            labels = ['Reference', 'Input', 'Synthetic target', 'Before', 'After',
                      'Known-theta oracle', 'Source shape GT', 'Source shape pred']
            for col, label in enumerate(labels):
                draw.text((col * 182 + 3, 8), label, fill='white')
            draw.text((3, 30), f"TRAIN ONLY {g['person']} / {g['mode']} | shape grayscale range 0.75..1.25; NOT final test", fill='white')
            for n in range(4):
                tensors = [g['extras']['reference'][n], g['source'][n], g['target'][n],
                           g['before'][n], p['loss_output'][n], oracle['loss_output'][n]]
                for col, tensor in enumerate(tensors):
                    canvas.paste(image(tensor), (col * 182 + 3, n * 205 + 65))
                for col, tensor in [(6, true_shape[n]), (7, predicted_shape[n])]:
                    canvas.paste(image(tensor, 0.75, 1.25), (col * 182 + 3, n * 205 + 65))
            canvas.save(output / f"{g['person']}_{g['mode']}_comparison.jpg")
            torch.save({'source_light_gt': source_true.cpu(),
                        'source_light_pred': p['source_illumination'].cpu(),
                        'target_light_gt': target_true.cpu(),
                        'target_light_pred': p['reference_illumination_on_source'].cpu(),
                        'gain_pred': p['transfer_gain'].cpu()},
                       output / f"{g['person']}_{g['mode']}_maps.pt")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--steps', type=int, default=600)
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--output-root', default='outputs/rrnet_fixed_group_diagnostic')
    args = parser.parse_args()
    if args.steps < 1:
        raise ValueError('steps must be positive')
    torch.manual_seed(20261001)
    np.random.seed(20261001)
    config = load_config(args.config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output = Path(args.output_root) / datetime.now().strftime('run_%Y%m%d_%H%M%S_%f')
    output.mkdir(parents=True, exist_ok=False)
    def log(message):
        line = f'[{datetime.now():%Y-%m-%d %H:%M:%S}] {message}'
        print(line, flush=True)
        with (output / 'train.log').open('a', encoding='utf-8') as handle:
            handle.write(line + '\n')
    log(f'Run directory: {output.resolve()} | FP32 | fixed BN | steps={args.steps}')
    model = ReferenceRelativeRRNet(**model_kwargs(config, args.config)).to(device).eval()
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint.get('model', checkpoint))
    criterion = ReferenceRelativeLoss(**config['loss']).to(device)
    d = config['data']
    ds = MEADPhysicalReferencePairs(
        d['root'], 'train', metadata_file=d.get('metadata_file', 'metadata.csv'),
        seed=20261001, variants_per_source=1, num_lights=config['model']['num_lights'],
        dynamic_epoch=False, reference_lighting_weights={'normal': 1},
        group_size=4, reference_sensitivity_fraction=0)
    people = sorted(set(ds.people))[:4]
    if len(people) < 4:
        raise ValueError('Need at least four distinct training identities')
    indices = [ds.people.index(person) for person in people]
    reference_item = ds[indices[3]]
    groups = []
    manifest = []
    # One fixed different-person reference; same four inputs paired with both
    # dark and normal reference lights prevents a constant-output solution.
    with torch.no_grad():
        for index, person in zip(indices[:3], people[:3]):
            original_item = ds[index]
            for mode in ('normal', 'dark'):
                item = copy.deepcopy(original_item)
                item['reference_clean'] = reference_item['source_clean'].clone()
                item['reference_relight_mask'] = reference_item['relight_mask'].clone()
                item['reference_mask'] = reference_item['source_light_mask'].clone()
                theta = sample_reference_theta(np.random.default_rng(
                    20261001 + int(mode == 'dark')), config['model']['num_lights'], mode)
                item['reference_theta_target'] = theta.unsqueeze(0).repeat(4, 1)
                item['reference_person_id'] = people[3]
                item['reference_lighting_mode'] = (mode,) * 4
                source, target, mask, skin, prediction, extras = model_prediction(
                    model, default_collate([item]), device, True, True, True)
                # Cached tensors are ordinary no-grad tensors, not inference-mode
                # tensors, so autograd can save them for parameter gradients.
                groups.append(dict(source=source, target=target, mask=mask,
                                   skin=skin, extras=extras, depth=prediction['depth'],
                                   reference_depth=prediction['reference_depth'],
                                   before=prediction['loss_output'].clone(),
                                   person=person, mode=mode))
                manifest.append(dict(source_person=person, source_index=index,
                                     source_clean_frame=ds.samples[index]['clean_frame'],
                                     reference_person=people[3],
                                     reference_clean_frame=ds.samples[indices[3]]['clean_frame'],
                                     mode=mode, source_theta=item['source_theta_target'].tolist(),
                                     reference_theta=theta.tolist()))
                log(f'Cached {person} / {mode}: four lights, shared exact target')
    (output / 'experiment.json').write_text(json.dumps(
        {'checkpoint': str(Path(args.checkpoint).resolve()), 'base_config': config,
         'steps': args.steps, 'learning_rate': args.learning_rate, 'seed': 20261001,
         'precision': 'FP32', 'fixed_bn': True, 'manifest': manifest,
         'scope': 'Training-only memorization diagnostic. No validation/test optimization.'},
        indent=2), encoding='utf-8')
    records = []
    def evaluate(step):
        rows = snapshot(model, criterion, groups)
        means = {key: sum(r['metrics'][key] for r in rows) / len(rows)
                 for key in rows[0]['metrics']}
        record = dict(step=step, average=means, groups=rows)
        records.append(record)
        with (output / 'metrics.jsonl').open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(record) + '\n')
        log(f"CHECK {step}/{args.steps} total={means['total']:.5f} "
            f"face_MAE={means['face_mae']:.5f} shape={means['illum_shape']:.5f} "
            f"gain={means['gain_log']:.5f} output_span={means['output_luma_span']:.5f}")
    evaluate(0)
    optimizer = torch.optim.Adam(model.base.lprm.parameters(), lr=args.learning_rate)
    start = time.monotonic()
    for step in range(1, args.steps + 1):
        model.train()
        for module in model.base.lprm.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()
        group = groups[(step - 1) % len(groups)]
        optimizer.zero_grad(set_to_none=True)
        loss = losses(criterion, group, forward(model, group))['total']
        if not torch.isfinite(loss):
            raise RuntimeError(f'Nonfinite loss at step {step}')
        loss.backward()
        optimizer.step()
        if step % 50 == 0:
            elapsed = time.monotonic() - start
            log(f'TRAIN {step}/{args.steps} loss={float(loss.detach()):.5f} '
                f'elapsed={elapsed:.0f}s ETA={elapsed / step * (args.steps - step):.0f}s')
        if step % 100 == 0 or step == args.steps:
            evaluate(step)
    torch.save({'model': model.state_dict(), 'config': config,
                'diagnostic_only': True, 'source_checkpoint': args.checkpoint,
                'step': args.steps}, output / 'diagnostic_final.pt')
    visualize(model, groups, output)
    (output / 'summary.json').write_text(json.dumps(
        {'before': records[0], 'after': records[-1], 'run_directory': str(output.resolve())},
        indent=2), encoding='utf-8')
    log('Completed. This checkpoint is diagnostic-only, not a replacement production model.')


if __name__ == '__main__':
    main()
