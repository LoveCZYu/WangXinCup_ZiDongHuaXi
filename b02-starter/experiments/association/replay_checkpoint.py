"""Replay a team checkpoint on explicitly selected labelled scenes."""
import argparse
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--scene', action='append', required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--enter', type=float, default=.50)
    parser.add_argument('--maintain', type=float, default=.15)
    args = parser.parse_args()
    if not 0 <= args.maintain <= args.enter <= 1:
        parser.error('Thresholds must satisfy 0 <= maintain <= enter <= 1')
    project = Path(__file__).resolve().parents[2]
    sys.path[:0] = [str(project), str(project/'experiments/point_scorer'),
                    str(project/'experiments/point_scorer_expanded')]

    import torch
    from b02.replay import replay
    from model import PointScorer
    from policy import Policy
    from experiments.association.run import make_handler

    torch.set_num_threads(4)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    model = PointScorer(temporal=True)
    model.load_state_dict(checkpoint['model'], strict=True)
    policy = Policy('replay', args.enter, args.maintain, 3)
    handler = make_handler(model.to(args.device).eval(), policy, .50, lambda: None)
    replay(args.data, args.out, handler,
           {'step': checkpoint['step'], 'seed': checkpoint['seed'], 'output_rule': vars(policy),
            'association': 'global_score', 'score_weight': .50}, scenes=args.scene)


if __name__ == '__main__':
    main()
