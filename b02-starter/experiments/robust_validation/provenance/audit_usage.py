"""Inspect existing provenance and scene names, without opening training CSVs."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import re

NAME=re.compile(r'^train_\d{3}$')
FIELDS={'fit','fit_scenes','train_scenes','training_scenes','train_files','training_files',
        'calibration','calibration_scenes','scenes','dev_scenes','validation_scenes','validation',
        'head_validation16','confirmation_scenes','evaluation_scenes','processed_scenes',
        'train','dev','train_specs','evaluation','train_names','dev_names','selected_train','selected_dev'}


def inspect(roots):
    observed=defaultdict(set);errors=[];files=0
    def record(value,path):
        if isinstance(value,str) and NAME.fullmatch(value):observed[value].add(str(path))
        elif isinstance(value,list):
            for item in value:
                record(item,path)
        elif isinstance(value,dict):
            for key in value:
                if NAME.fullmatch(key):observed[key].add(str(path))
                elif isinstance(value[key],(dict,list)):record(value[key],path)
    def walk(value,path,context=''):
        if isinstance(value,dict):
            for key,item in value.items():
                if key in FIELDS and context not in ('split','splits','roles','data_roles'):
                    record(item,path)
                # Per-target labels, model statistics and frozen split copies are not needed.
                if key not in ('per_target','split','splits','roles','data_roles','source_sha256','runtime_sha256'):
                    walk(item,path,key)
        elif isinstance(value,list):
            for item in value:
                if isinstance(item,(dict,list)):walk(item,path,context)
    for root in roots:
        for path in sorted(root.rglob('*')):
            relative=path.relative_to(root)
            if any(p.startswith('source') or p in ('__pycache__','site-packages','.git') for p in relative.parts[:-1]):continue
            if not path.is_file():continue
            if NAME.fullmatch(path.stem) and path.suffix in ('.json','.jsonl','.html','.npz','.h5'):
                observed[path.stem].add(str(path))
            if path.suffix=='.json':
                files+=1
                try:walk(json.loads(path.read_text(encoding='utf-8-sig')),path)
                except (ValueError,UnicodeError) as error:errors.append({'path':str(path),'error':type(error).__name__})
    return {'files_checked':files,'observed':{name:sorted(paths) for name,paths in sorted(observed.items())},'errors':errors}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',nargs='+',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();result=inspect(args.root)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('metadata files',result['files_checked'],'observed scene names',len(result['observed']),'parse errors',len(result['errors']))


if __name__=='__main__':main()
