from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path

root=Path('b02-starter');work=Path('analysis/validation-work')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
audit=json.loads((work/'data-audit.json').read_text(encoding='utf-8'))
for name in ('usage-local-v2.json','usage-cloud-v2.json'):
    usage=json.loads((work/name).read_text(encoding='utf-8'))
    assert not usage['errors'] and not set(audit['work_validation16'])&set(usage['observed'])
audit['supplemental_audit']={name:sha(work/name) for name in ('usage-local-v2.json','usage-cloud-v2.json')}
(work/'data-audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
paths={'point':root/'assets/point-scorer-8000.pt','prior':root/'runs/v3-tracking/memory-selected.pt',
       'affinity':root/'runs/v2-id-recovery/affinity.pt','head':root/'runs/v2-continuation/training/20261116.pt',
       'fusion':root/'runs/v2-continuation/fusion-calibration/20261116.json'}
sources=['b02/algorithm.py','b02/point_tracker.py','experiments/point_scorer/model.py',
         'experiments/point_scorer_expanded/policy.py','experiments/association/interventions.py','experiments/association/assignment.py',
         *(f'experiments/v2_association/{name}.py' for name in ('features','context','model','tracker')),
         'experiments/v3_tracking/memory.py','experiments/v3_tracking/tracker.py','experiments/v3_tracking/runtime.py',
         'experiments/id_recovery/affinity.py','experiments/id_recovery/tracker.py','experiments/prebirth/tracker.py',
         *(f'experiments/continuation/{name}.py' for name in ('tracker','learn','fusion','memory_guard')),
         'profiles/current.json','b02/scorecard.py']
freeze={'format':'b02-frozen-work-validation-v1','frozen_at_utc':datetime.now(timezone.utc).isoformat(),
        'source_commit':'bfeaed6','modes':['wide','memory','candidate'],'primary_comparison':'candidate minus wide',
        'secondary':'memory as a fixed conservative comparison; no post-validation winner selection',
        'artifacts':{name:sha(path) for name,path in paths.items()},'sources':{name:sha(root/name) for name in sources},
        'calibration_reference':{'wide':'v2-continuation/wide_shadow','memory':'v2-continuation/memory_guard','candidate':'v2-continuation/fusion_memory-20261116'},
        'work_validation16':audit['work_validation16'],'data_sha256':audit['archive_sha256'],'data_audit_sha256':sha(work/'data-audit.json'),
        'bootstrap':{'unit':'paired scenes','resamples':10000,'seed':20261116,'interval':'percentile 95%; scene-independence assumption not guaranteed'},
        'adoption':{'mixed_rate_ratio_max':.95,'mixed_rate_difference_ci_upper_max':0.,'weighted_delta_ci_lower_min':-.15,
                    'point_recall_drop_pp_max':.1,'track_recall_drop_pp_max':.1,'continuous_drop_max':.3,'timeliness_drop_max':.3,
                    'clutter_ratio_max':1.02,'any_scene_total_drop_max':1.,'any_scene_point_drop_pp_max':1.},
        'decision_rule':'support if all adoption conditions pass; regression if an aggregate guard or worst-scene guard fails, or CI establishes worse identity/score; otherwise insufficient evidence',
        'after_results':'Mark these 16 as exposed work validation. No threshold or checkpoint changes in this stage.',
        'sealed_test_used':False,'confirmation16_used':False,'new_training':False}
destination=root/'experiments/robust_validation/freeze.json'
assert not destination.exists()
destination.write_text(json.dumps(freeze,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print('Frozen candidates, source hashes, 16 scenes and adoption rules:',sha(destination))
