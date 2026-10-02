"""Quantify gain bounds and spatial errors for a completed fixed-group diagnostic."""
import argparse
import json
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rrnet.config import model_kwargs
from rrnet.losses import luminance
from rrnet.physical_reference_data import MEADPhysicalReferencePairs
from rrnet.reference_relative_model import ReferenceRelativeRRNet


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True)
    args = parser.parse_args()
    run = Path(args.run)
    experiment = json.loads((run / 'experiment.json').read_text(encoding='utf-8'))
    config = experiment['base_config']
    # CPU renderer/compositor only: saved illumination maps, no model inference.
    model = ReferenceRelativeRRNet(**model_kwargs(
        config, 'configs/rrnet_bn_diagnostic_fixed_200.yaml')).eval()
    d = config['data']
    ds = MEADPhysicalReferencePairs(
        d['root'], 'train', metadata_file=d.get('metadata_file', 'metadata.csv'),
        seed=20261001, variants_per_source=1, num_lights=config['model']['num_lights'],
        dynamic_epoch=False, reference_lighting_weights={'normal': 1}, group_size=4)
    rows = []
    with torch.no_grad():
        for entry in experiment['manifest']:
            person, mode = entry['source_person'], entry['mode']
            item = ds[entry['source_index']]
            maps = torch.load(run / f'{person}_{mode}_maps.pt', weights_only=False)
            clean, mask = item['source_clean'], item['relight_mask']
            face = item['source_light_mask']
            source = model.renderer.blend_relight(
                clean * maps['source_light_gt'], clean, mask).clamp(0, 0.999)
            target = model.renderer.blend_relight(
                clean * maps['target_light_gt'], clean, mask).clamp(0, 0.999)
            raw_gain = (maps['target_light_gt']
                        / maps['source_light_gt'].clamp_min(1e-8))
            gain = model.compute_transfer_gain(source, maps['source_light_gt'],
                                               maps['target_light_gt'])
            oracle = model.renderer.blend_relight(source * gain, source, mask)
            def mean(x):
                return (x * face).sum((1, 2, 3)) / (
                    x.shape[1] * face.sum((1, 2, 3)).clamp_min(1))
            def normalized(x):
                x = luminance(x)
                return x / mean(x)[:, None, None, None].clamp_min(1e-4)
            predicted = model.renderer.blend_relight(
                source * maps['gain_pred'], source, mask)
            rows.append(dict(person=person, mode=mode,
                oracle_face_mae=float(mean((oracle-target).abs()).mean()),
                true_gain_below_min_fraction=float(mean((raw_gain < model.min_transfer_gain).float()).mean()),
                true_gain_above_max_fraction=float(mean((raw_gain > model.max_transfer_gain).float()).mean()),
                predicted_source_shape_mae=float(mean((normalized(maps['source_light_pred'])-normalized(maps['source_light_gt'])).abs()).mean()),
                output_shape_mae=float(mean((normalized(predicted)-normalized(target)).abs()).mean()),
                per_input_face_mae=mean((predicted-target).abs()).tolist(),
                output_face_luma=mean(luminance(predicted)).tolist(),
                target_face_luma=mean(luminance(target)).tolist()))
    output = run / 'gain_bounds_audit.json'
    output.write_text(json.dumps({'rows': rows, 'scope': 'Training-only diagnostic'}, indent=2), encoding='utf-8')
    print(json.dumps(rows), flush=True)


if __name__ == '__main__':
    main()
