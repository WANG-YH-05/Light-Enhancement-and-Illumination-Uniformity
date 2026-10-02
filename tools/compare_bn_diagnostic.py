"""Visual comparison on identical held-out physical groups, not a final test."""
import argparse
import json
import sys
from pathlib import Path

import torch
from PIL import Image, ImageDraw
from torch.utils.data import default_collate

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rrnet.config import load_config, model_kwargs
from rrnet.physical_reference_data import MEADPhysicalReferencePairs
from rrnet.reference_relative_model import ReferenceRelativeRRNet
from train import model_prediction


def tile(tensor):
    pixels = (tensor.detach().float().clamp(0, 1).permute(1, 2, 0)
              .cpu().numpy() * 255).round().astype('uint8')
    return Image.fromarray(pixels).resize((200, 200), Image.Resampling.LANCZOS)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--original', required=True)
    parser.add_argument('--fixed', required=True)
    parser.add_argument('--baseline', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--fixed-label', default='Fixed BN +200')
    parser.add_argument('--baseline-label', default='Normal BN +200')
    args = parser.parse_args()
    config = load_config(args.config)
    d = config['data']
    dataset = MEADPhysicalReferencePairs(
        d['root'], 'val', metadata_file=d.get('metadata_file', 'metadata.csv'),
        seed=d['sampling_seed'] + 1, variants_per_source=1,
        num_lights=config['model']['num_lights'], dynamic_epoch=False,
        reference_lighting_weights=d['reference_lighting_weights'],
        group_size=4, reference_sensitivity_fraction=0.0)
    indices = [dataset.people.index(p) for p in sorted(set(dataset.people))]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ReferenceRelativeRRNet(**model_kwargs(config, args.config)).to(device).eval()
    # Cache the same three groups for all checkpoints. FP32 matches validation.
    batches = [default_collate([dataset[i]]) for i in indices]
    records = []
    columns = []
    with torch.inference_mode():
        for label in ('original', 'fixed', 'baseline'):
            state = torch.load(getattr(args, label), map_location='cpu', weights_only=False)
            model.load_state_dict(state.get('model', state))
            model.eval()
            images = []
            for index, batch in zip(indices, batches):
                source, target, mask, skin, pred, extras = model_prediction(
                    model, batch, device, True, True, True)
                if label == 'original':
                    records.append((dataset.people[index],
                                    [tile(x) for x in extras['reference']],
                                    [tile(x) for x in source],
                                    [tile(x) for x in target]))
                images.extend(tile(x) for x in pred['loss_output'])
            columns.append(images)
            print('Compared:', label, flush=True)
    canvas = Image.new('RGB', (1240, 12 * 235 + 70), '#161616')
    draw = ImageDraw.Draw(canvas)
    labels = ['Reference', 'Input', 'Synthetic target', 'Original 8000', args.fixed_label, args.baseline_label]
    for c, label in enumerate(labels):
        draw.text((c * 205 + 5, 8), label, fill='white')
    draw.text((5, 30), 'Same held-out inputs / reference. Targets are synthetic, not measured lighting ground truth.', fill='white')
    row = 0
    for person, refs, sources, targets in records:
        for n in range(4):
            y = 70 + row * 235
            draw.text((5, y - 15), f'{person} / input light {n + 1}', fill='white')
            cells = [refs[n], sources[n], targets[n]] + [column[row] for column in columns]
            for c, cell in enumerate(cells):
                canvas.paste(cell, (c * 205 + 5, y))
            row += 1
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)
    output.with_suffix('.json').write_text(json.dumps(
        {'config': args.config, 'original': args.original, 'fixed': args.fixed,
         'baseline': args.baseline, 'validation_indices': indices,
         'fixed_label': args.fixed_label, 'baseline_label': args.baseline_label,
         'scope': 'Three held-out identities, four source lights each; visual diagnostic only.'},
        indent=2), encoding='utf-8')
    print(output.resolve(), flush=True)


if __name__ == '__main__':
    main()
