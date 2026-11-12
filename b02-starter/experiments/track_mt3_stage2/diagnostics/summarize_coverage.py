"""Summarize prespecified coverage comparisons without selecting checkpoints."""
import argparse
import json
from pathlib import Path


def summarize(report):
    manifest = report['manifest']
    index = {}
    for row in report['evaluations']:
        key = (row['count'], row['step'])
        if key in index:
            raise ValueError(f'Duplicate evaluation: {key}')
        index[key] = row
    comparisons = {}
    for name in ('equal_total_updates', 'equal_exposures_per_clip'):
        rows = []
        for count in manifest['counts']:
            step = manifest[name] * (count if name == 'equal_exposures_per_clip' else 1)
            evaluation = index.get((count, step))
            if evaluation is None:
                rows.append({'count': count, 'step': step, 'available': False})
                continue
            rows.append({
                'count': count, 'step': step, 'available': True,
                'exposures_per_clip': evaluation['exposures_per_clip'],
                'metrics': evaluation['metrics'],
            })
        comparisons[name] = rows
    return {
        'complete': report.get('complete', False),
        'elapsed_seconds': report.get('elapsed_seconds'),
        'actual_updates': sum(max((step for n, step in index if n == count), default=0)
                              for count in manifest['counts']),
        'peak_allocated_gib': max(row['peak_allocated_gib'] for row in index.values()),
        'note': 'Single-seed diagnostic, not an official score. Same-file evaluation sets differ by arm.',
        **comparisons,
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    args = parser.parse_args()
    result = summarize(json.loads((args.run/'report.json').read_text(encoding='utf-8')))
    (args.run/'summary.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items() if not isinstance(v, list)}))
    for name in ('equal_total_updates', 'equal_exposures_per_clip'):
        print(name)
        for row in result[name]:
            if not row['available']:
                print(f"  clips={row['count']} step={row['step']}: unavailable")
                continue
            m = row['metrics']
            print(f"  clips={row['count']} step={row['step']} "
                  f"seen={m['seen_free']['recall']:.4%} oracle={m['seen_oracle']['recall']:.4%} "
                  f"geometry={m['seen_free']['geometry_recall']:.4%} "
                  f"external={m['different_file']['recall']:.4%} "
                  f"external_precision={m['different_file']['precision']:.4%} "
                  f"external_f1={m['different_file']['point_f1']:.4%}")
