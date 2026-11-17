"""Fixed 2x2: missing-candidate backup and existing column identity evidence."""
from experiments.candidate_onset.fallback import install as backup
from experiments.identity_competition.scoring import ColumnScorer

MODES=('baseline','column','backup','combined')


def install(tracker,mode):
    assert mode in MODES
    backup(tracker,'backup_guarded' if mode in ('backup','combined') else 'baseline')
    if mode in ('column','combined'):
        tracker.model_scorer=ColumnScorer(tracker.model_scorer,tracker.affinity,tracker,'identity')
        tracker.column_evidence=tracker.model_scorer
    return tracker
