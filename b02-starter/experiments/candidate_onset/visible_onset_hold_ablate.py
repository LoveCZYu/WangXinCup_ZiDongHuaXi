"""Disable one observed hold in the frozen automatic work replay."""
import argparse
import json
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np

from experiments.candidate_onset import visible_onset_hold as experiment
from experiments.candidate_onset.joint_event import trace_keys


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--skip-circle', type=int, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    probe, remaining = parser.parse_known_args()
    traces = [json.loads(line) for line in probe.reference.read_text(encoding='utf8').splitlines()]
    expected = {row['circle']: trace_keys(row['tracks']) for row in traces if row['circle'] <= probe.skip_circle}
    original_run, original_score = experiment.run, experiment.score

    def replay(frames, cached, tracker, mode, model, stream):
        if mode != 'automatic':
            # The caller's standard packaging includes this mode. Only the
            # automatic scored replay is an arm of this single-action probe.
            return dict(identity={'mixed_minority_truth_points': None}, operations={})
        current = None

        def cache(i):
            nonlocal current
            current = frames[i][0]
            return cached(i)

        def classify(model, values):
            if current == probe.skip_circle:
                return np.full(len(values), -np.inf)
            return original_score(model, values)

        class Checked:
            def write(self, line):
                row = json.loads(line)
                if row['circle'] in expected:
                    assert trace_keys(row['tracks']) == expected.pop(row['circle'])
                stream.write(line)

        with patch.object(experiment, 'score', classify):
            result = original_run(frames,cache,tracker,mode,model,Checked())
        assert not expected and all(action['circle'] != probe.skip_circle for action in result['actions'])
        result['disabled_circle'] = probe.skip_circle
        print(probe.skip_circle,result['identity']['mixed_minority_truth_points'],result['continued'],flush=True)
        return result

    with patch.object(sys, 'argv', [sys.argv[0],*remaining]), patch.object(experiment,'run',replay):
        experiment.main()


if __name__ == '__main__':
    main()
