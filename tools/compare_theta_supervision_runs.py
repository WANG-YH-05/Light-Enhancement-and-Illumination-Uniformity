"""Compare theta-weight ablations without confusing differently weighted totals."""
import argparse
import json
from pathlib import Path


def load_rows(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--control-run', required=True)
    parser.add_argument('--weak-run', required=True)
    parser.add_argument('--initial-report', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    initial = json.loads(Path(args.initial_report).read_text(encoding='utf-8'))
    rows = []
    for label, directory, theta_weight in (
            ('control', args.control_run, 0.5), ('weak', args.weak_run, 0.1)):
        for row in load_rows(Path(directory) / 'val_metrics.jsonl'):
            record = dict(row, variant=label,
                          checkpoint=str((Path(directory) / f"rrnet_step_{row['step']:07d}.pt").resolve()))
            record['common_total'] = row['total'] + (0.5 - theta_weight) * (
                row['theta_source'] + row['theta_reference'])
            rows.append(record)
    report = {'initial': initial, 'rows': rows,
              'best_relight': {label: min((r for r in rows if r['variant'] == label),
                                         key=lambda r: r['relight']) for label in ('control', 'weak')},
              'notes': ['Same deterministic 30-group validation; not untouched final test.',
                        'Common total uses theta weights 0.5/0.5; raw totals are not comparable.',
                        'Short single-seed ablation; no statistical significance claim.']}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    for row in rows:
        print(json.dumps({k: row[k] for k in ('variant', 'step', 'common_total', 'relight',
                                            'gain_log', 'illum_shape', 'consistency',
                                            'shape_consistency', 'reference_contrast')}))
    print('Best relight checkpoints:', {label: row['checkpoint'] for label, row in report['best_relight'].items()})


if __name__ == '__main__':
    main()
