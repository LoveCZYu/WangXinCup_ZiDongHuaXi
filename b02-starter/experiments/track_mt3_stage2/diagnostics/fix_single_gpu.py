"""Apply the single-GPU scheduling correction to an existing isolated fork."""
import argparse
import difflib
from pathlib import Path
p=argparse.ArgumentParser()
p.add_argument('upstream',type=Path)
a=p.parse_args()
path=a.upstream/'track_mt3/models/track_mt3.py'
before=path.read_text(encoding='utf-8')
old='guaranteed = index == 0 and time_index < 2'
new='guaranteed = torch.distributed.is_initialized() and index == 0 and time_index < 2'
assert before.count(old)==1, 'Expected original single-GPU override exactly once'
after=before.replace(old,new)
path.write_text(after,encoding='utf-8')
with (a.upstream/'b02-stage2.patch').open('a',encoding='utf-8') as file:
    file.writelines(difflib.unified_diff(before.splitlines(True),after.splitlines(True),
                                      fromfile='models/track_mt3.py',tofile='models/track_mt3.py'))
print('Corrected single-GPU teacher-forcing scheduling; inference code unchanged.')
