"""Separate each affected ID's observation history from its context history."""
from copy import deepcopy
from unittest.mock import patch

from experiments.candidate_onset import joint_event

MODES = tuple(f'{side}_{field}' for side in ('donor', 'receiver') for field in ('points', 'context', 'both'))


def replace_short(tracker, control, pair, mode):
    side, field = mode.split('_')
    batch = pair[side == 'receiver']
    fields = dict(points=('history',), context=('context_history',), both=joint_event.GROUPS['short'])
    for name in fields[field]:
        source, destination = getattr(control, name), getattr(tracker, name)
        destination[batch] = deepcopy(source[batch])


if __name__ == '__main__':
    with patch.object(joint_event, 'MODES', MODES), patch.object(joint_event, 'replace_state', replace_short):
        joint_event.main()
