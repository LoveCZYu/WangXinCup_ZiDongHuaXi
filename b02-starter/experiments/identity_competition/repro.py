"""Replay only project-created, SHA-checked private diagnostic fixtures."""
import argparse
import hashlib
import json
from pathlib import Path
import pickle
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def replay(path):
    manifest = json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))
    raw = path.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == manifest['fixture_sha256']
    # These files are written by capture.py in our private run directory.
    # Do not use this loader on third-party or user-supplied pickle files.
    packet = pickle.loads(raw)
    tracker = pickle.loads(packet['before'])
    prediction = tracker(packet['points'], packet['scores'])
    return manifest, packet, tracker, prediction


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('fixture', type=Path)
    parser.add_argument('--mode', choices=('baseline', 'identity', 'motion', 'combined', 'volume', 'identity_volume'), default='baseline')
    args = parser.parse_args()
    import torch
    torch.set_num_threads(1)
    manifest, packet, tracker, output = replay(args.fixture)
    assert output == packet['observed_outputs'], 'Captured original step is not reproducible'
    again = replay(args.fixture)[-1]
    assert output == again, 'Nondeterministic original step'
    original = output
    if args.mode != 'baseline':
        from experiments.identity_competition.scoring import install
        output = install(pickle.loads(packet['before']), args.mode)(packet['points'], packet['scores'])
        again = install(pickle.loads(packet['before']), args.mode)(packet['points'], packet['scores'])
        assert output == again, 'Nondeterministic counterfactual step'
    actual = next((p['BatchID'] for p in output if p['PntID'] == manifest['point_id']), None)
    passed = actual == manifest['expected_previous_id']
    print(json.dumps({'symptom': 'published target changes ID despite old candidate edge',
        'scene': manifest['scene'], 'circle': manifest['circle'], 'point': manifest['point_id'],
        'expected_previous_id': manifest['expected_previous_id'], 'actual_id': actual,
        'mode': args.mode, 'selected_points_unchanged': {p['PntID'] for p in output} == {p['PntID'] for p in original},
        'repeat_exact': True, 'verdict': 'GREEN' if passed else 'RED'}, ensure_ascii=False))
    raise SystemExit(0 if passed else 1)


if __name__ == '__main__':
    main()
