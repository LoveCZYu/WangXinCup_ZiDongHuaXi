"""Pack only the planned train/calibration/dev scenes from the official archive."""
import argparse
import json
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('data', type=Path)
p.add_argument('out', type=Path)
p.add_argument('--project', type=Path, default=Path('.'))
p.add_argument('--plan', type=Path, default=Path(__file__).with_name('plan.json'))
a = p.parse_args()
plan = json.loads(a.plan.read_text())
split = json.loads((a.project/'splits/v1.json').read_text())
fit, cal, dev = (set(plan[k]) for k in ('fit_scenes', 'calibration_scenes', 'dev_scenes'))
assert fit.isdisjoint(cal) and fit | cal <= set(split['train'])
assert dev == set(split['dev']) and not (fit | cal | dev) & set(split['holdout'])
selected = fit | cal | dev
with ZipFile(a.data) as source:
    members = [entry for entry in source.infolist()
               if entry.filename.lower().endswith('.csv') and Path(entry.filename).stem in selected]
    assert len(members) == len(selected) and {Path(e.filename).stem for e in members} == selected
    with ZipFile(a.out, 'x', ZIP_DEFLATED, compresslevel=1) as target:
        for entry in members:
            target.writestr(Path(entry.filename).name, source.read(entry))
print(f'Packed {len(selected)} scenes, excluding holdout: {a.out}')
