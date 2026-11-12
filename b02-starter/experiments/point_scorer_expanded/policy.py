"""Causal output-only filtering; baseline associations and IDs remain untouched."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Policy:
    name: str
    enter: float
    maintain: float
    grace: int = 0


POLICIES = [Policy(f'fixed_{value:g}', value, value) for value in (.1, .15, .25, .35, .5)] + [
    Policy('history_05_015', .5, .15, 3),
    Policy('history_035_01', .35, .1, 3),
]


class OutputGate:
    def __init__(self, policy):
        self.policy = policy
        self.last_strong = {}

    def apply(self, outputs, scores, circle):
        p = self.policy
        self.last_strong = {batch: t for batch, t in self.last_strong.items() if circle-t <= p.grace}
        result = []
        for point in outputs:
            score = scores[point['PntID']]
            batch = point['BatchID']
            strong = score >= p.enter
            recent = batch in self.last_strong and circle-self.last_strong[batch] <= p.grace
            if strong or (p.grace > 0 and recent and score >= p.maintain):
                result.append(point)
            if strong and p.grace > 0:
                self.last_strong[batch] = circle
        return result


def check():
    point = {'PntID': 10, 'BatchID': 7}
    gate = OutputGate(Policy('test', .5, .15, 3))
    assert not gate.apply([point], {10: .2}, 0)
    assert gate.apply([point], {10: .8}, 1)[0] is point
    assert gate.apply([point], {10: .2}, 4)[0] is point
    assert not gate.apply([point], {10: .2}, 5)
    assert not OutputGate(Policy('test', .5, .15, 3)).apply([point], {10: .2}, 2)
    gate = OutputGate(Policy('test', .5, .15, 3))
    gate.apply([point], {10: .8}, 0)
    assert not gate.apply([{'PntID': 10, 'BatchID': 8}], {10: .2}, 1)
    assert not gate.apply([point], {10: .1}, 1)
    print('Output gate: no unconfirmed low-score birth, bounded grace, ID separation and identity preservation passed')


if __name__ == '__main__':
    check()
