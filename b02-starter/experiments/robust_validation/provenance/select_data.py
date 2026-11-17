"""Select work validation using provenance and unlabelled input-prefix overlap."""
import csv
import hashlib
from io import TextIOWrapper
import json
from pathlib import Path,PurePosixPath
import re
from struct import Struct
from zipfile import ZipFile,ZIP_DEFLATED

ROOT=Path('b02-starter')
OUT=Path('analysis/validation-work')
plan=json.loads((ROOT/'splits/v3_tracking.json').read_text(encoding='utf-8'))
observed=set()
evidence={}
for name in ('usage-local.json','usage-cloud.json'):
    data=json.loads((OUT/name).read_text(encoding='utf-8'))
    assert not data['errors'],data['errors']
    observed.update(data['observed'])
    evidence[name]={'metadata_files':data['files_checked'],'scene_names':sorted(data['observed'])}
excluded=set(plan['fit64']+plan['calibration']+plan['confirmation16']+plan['development_regression']+plan['sealed_excluded'])
pool=sorted(set(plan['fit116'])-excluded)
eligible=sorted(set(pool)-observed)
assert len(pool)==52 and len(eligible)>=16
source=Path('智能体应用赛题B02--目标检测跟踪--初赛训练数据集.zip')
archive_sha=hashlib.file_digest(source.open('rb'),'sha256').hexdigest()
prefixes={};metadata={};token=Struct('<Iqqq')
with ZipFile(source) as archive:
    index={PurePosixPath(n).stem:n for n in archive.namelist() if re.fullmatch(r'train_\d{3}\.csv',PurePosixPath(n).name)}
    assert len(index)==200
    # No sealed CSV is opened, including the historical train_003 exclusion.
    screen=sorted((observed|set(eligible))-set(plan['sealed_excluded']))
    for scene in screen:
        with archive.open(index[scene]) as raw,TextIOWrapper(raw,encoding='utf-8-sig',newline='') as text:
            rows=csv.reader(text);header=next(rows)
            columns=[header.index(k) for k in ('Pt_Circle','Pt_Azi','Pt_El','Pt_Distance')]
            first=None;circles=set();values=set()
            for row in rows:
                circle=int(float(row[columns[0]]))
                if first is None:first=circle
                if circle not in circles and len(circles)>=64:break
                circles.add(circle)
                xyz=[round(float(row[k])*10000) for k in columns[1:]]
                values.add(token.pack(circle-first,*xyz))
        prefixes[scene]=values
        metadata[scene]={'prefix_circles':len(circles),'quantized_points':len(values),
                         'prefix_sha256':hashlib.sha256(b''.join(sorted(values))).hexdigest(),
                         'zip_crc32':archive.getinfo(index[scene]).CRC,'csv_bytes':archive.getinfo(index[scene]).file_size}
    blocked={};ranked=sorted(eligible,key=lambda s:hashlib.sha256(('b02-work-validation-20261116:'+s).encode()).hexdigest())
    screened=[];overlap={}
    for scene in ranked:
        values=prefixes[scene]
        if len(values)<100:
            blocked[scene]='Insufficient unlabelled prefix';continue
        best=('none',0.)
        for prior in sorted((observed-set(plan['sealed_excluded']))|set(screened)):
            if prior not in prefixes:continue
            other=prefixes[prior]
            # Jaccard cannot reach .8 if the set sizes differ by more than 20%.
            if min(len(values),len(other))/max(1,max(len(values),len(other)))<.8:continue
            common=len(values&other);score=common/max(1,len(values)+len(other)-common)
            if score>best[1]:best=(prior,score)
        overlap[scene]={'nearest_prefix':best[0],'jaccard':best[1]}
        if best[1]>=.8:blocked[scene]='High unlabelled-prefix overlap with '+best[0]
        else:screened.append(scene)
    selected=screened[:16]
    assert len(selected)==16 and not set(selected)&excluded and not set(selected)&observed
    out=OUT/'data-new-work16.zip'
    with ZipFile(out,'x',ZIP_DEFLATED) as target:
        for scene in selected:target.writestr(scene+'.csv',archive.read(index[scene]))
result={'format':'b02-work-validation-data-audit-v1','scope':'current model lineage and available local/cloud experiment metadata',
        'source_archive_sha256':archive_sha,'pool52':pool,'previously_observed_in_pool':sorted(set(pool)&observed),
        'metadata_evidence':evidence,'selection_rule':'SHA256(b02-work-validation-20261116:scene), first 16 after exclusions and unlabelled screening',
        'work_validation16':selected,'remaining_candidates':screened[16:],'blocked':blocked,
        'unlabelled_screen':{'circles':64,'geometry_quantization':.0001,'overlap_reject':.8,'labels_used':False,
                             'limitations':'High input-prefix overlap screening only; cannot rule out shared simulated target trajectories or common generators',
                             'metadata':metadata,'candidate_overlap':overlap},
        'archive_sha256':hashlib.file_digest(out.open('rb'),'sha256').hexdigest(),'archive_bytes':out.stat().st_size,
        'sealed_csv_opened':False,'performance_or_labels_viewed_for_selection':False,
        'houyi':'Not an ancestor of frozen current-route weights; other-route record unavailable and no cross-route unseen-data claim made'}
(OUT/'data-audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'pool':len(pool),'metadata_used_excluded':len(set(pool)&observed),'blocked':blocked,
                  'selected':selected,'archive_bytes':out.stat().st_size,'archive_sha256':result['archive_sha256']},ensure_ascii=False))
