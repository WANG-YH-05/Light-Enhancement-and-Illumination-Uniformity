"""Compare matched affine/smooth lighting diagnostics, not final test metrics."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rrnet.config import model_kwargs
from rrnet.losses import luminance
from rrnet.reference_relative_model import ReferenceRelativeRRNet
from tools.train_lprm_smooth_pilot import cache, objective


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--smooth', required=True)
    p.add_argument('--affine', required=True)
    a = p.parse_args()
    torch.set_num_threads(4)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    models, configs = {}, {}
    for name in ('smooth', 'affine'):
        checkpoint = torch.load(getattr(a, name), map_location='cpu', weights_only=False)
        c = checkpoint['config']
        # Resolution base for relative paths is the project's configs folder.
        model = ReferenceRelativeRRNet(**model_kwargs(c, 'configs/rrnet_lprm_smooth_pilot.yaml')).to(device).eval()
        model.load_state_dict(checkpoint['model'], strict=True)
        models[name], configs[name] = model, c
    if configs['smooth']['data'] != configs['affine']['data']:
        raise ValueError('Not a matched data comparison')
    groups, manifest = cache(models['smooth'], configs['smooth'], 'val',
                             configs['smooth']['diagnostic']['validation_groups_per_person'], device)
    output = Path('outputs/rrnet_lprm_decoder_comparison') / datetime.now().strftime('run_%Y%m%d_%H%M%S_%f')
    output.mkdir(parents=True, exist_ok=False)
    canvas = Image.new('RGB', (5 * 200, len(groups) * 225 + 50), '#151515')
    draw = ImageDraw.Draw(canvas)
    labels = ['LPRM gray input', 'Known light shape', 'Affine shape', 'Smooth shape', 'Known light magnitude']
    for i, label in enumerate(labels):
        draw.text((i * 200 + 4, 8), label, fill='white')
    draw.text((4, 28), 'VAL diagnostic | shapes 0.75..1.25 | magnitude 0..1.2 | NO image residual/AGM', fill='white')
    records = []
    def picture(x, low=0., high=1.):
        x = ((x.detach() - low) / (high - low)).clamp(0, 1)
        if x.shape[0] == 1:
            x = x.expand(3, -1, -1)
        pixels = (x.permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)
        return Image.fromarray(pixels).resize((192, 192), Image.Resampling.LANCZOS)
    with torch.no_grad():
        for row, (group, info) in enumerate(zip(groups, manifest)):
            group = {k: v.to(device) for k, v in group.items()}
            mask = group['mask']
            def shape(x):
                mean = (x * mask).sum((2, 3), keepdim=True) / mask.sum((2, 3), keepdim=True).clamp_min(1)
                return x / mean.clamp_min(1e-4)
            gt_shape = shape(group['light'])
            n_source = 4 if info['uniform'] else 1
            # Explicitly select the strongest source spatial variation, not a
            # reference or a random trivial uniform-light member.
            spread = ((gt_shape[:n_source] - 1).square() * mask[:n_source]).sum((1, 2, 3))
            index = int(spread.argmax())
            predicted, metrics = {}, {}
            for name, model in models.items():
                pred = model.base.lprm(group['input'])
                light, _ = model.illumination_from_theta(group['depth'], pred['theta'])
                predicted[name] = shape(luminance(light))
                metrics[name] = {k: float(v) for k, v in objective(model, group, configs[name]['diagnostic']).items()}
                color = pred['theta'][:, :-3].reshape(-1, 9, 10)[..., :3]
                metrics[name]['negative_color_fraction'] = float((color < 0).float().mean())
                metrics[name]['fully_off_light_fraction'] = float((color.clamp_min(0).sum(-1) == 0).float().mean())
            records.append(dict(**info, metrics=metrics, displayed_member=index))
            for column, (tensor, lo, hi) in enumerate([
                    (group['input'][index], 0, 1), (gt_shape[index], .75, 1.25),
                    (predicted['affine'][index], .75, 1.25),
                    (predicted['smooth'][index], .75, 1.25), (group['light'][index], 0, 1.2)]):
                canvas.paste(picture(tensor, lo, hi), (column * 200 + 4, row * 225 + 50))
            draw.text((4, row * 225 + 245), info['source_person'] + f' / variant {row % 2}', fill='white')
    averages = {name: {k: sum(r['metrics'][name][k] for r in records) / len(records)
                       for k in records[0]['metrics'][name]} for name in models}
    canvas.save(output / 'validation_light_shapes.jpg')
    report = dict(checkpoints=vars(a), averages=averages, rows=records,
                  scope='Six balanced validation groups; small cached estimator diagnostic; not untouched test or intrinsic albedo ground truth')
    (output / 'comparison.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(averages, indent=2), flush=True)
    print(output.resolve(), flush=True)


if __name__ == '__main__':
    main()
