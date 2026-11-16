"""Prevent wrong inputs, mismatched code, or extra files in an A-board upload."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from zipfile import ZipFile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
from build_results import build, EXPECTED


def write_json(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value),encoding='utf-8')


class ResultPackageTest(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base=Path(self.temp.name)
        self.inputs=self.base/'inputs';self.inputs.mkdir()
        self.first=self.base/'first';self.second=self.base/'second'
        self.code=self.base/'code.zip';self.out=self.base/'results.zip';self.report=self.base/'report.json'
        self.policy=self.base/'policy.json'
        manifest={'team_id':'T000906','version':'v1.1',
                  'files':{'src/main.py':hashlib.sha256(b'# fixture only').hexdigest()}}
        raw=json.dumps(manifest).encode()
        with ZipFile(self.code,'w') as z:
            z.writestr('PACKAGE.json',raw);z.writestr('src/main.py',b'# fixture only')
        manifest_sha=hashlib.sha256(raw).hexdigest()
        policy={'date':'2026-11-16','result_zip_files':sorted(EXPECTED),'inputs':{},
                'historical_holdout_overlap':['train_003'],'remaining_independent_holdout':['unseen']}
        packets=[]
        for i,name in enumerate(sorted(EXPECTED),1):
            data=f'synthetic input {i}'.encode();(self.inputs/name).write_bytes(data)
            policy['inputs'][name]={'sha256':hashlib.sha256(data).hexdigest()}
            packets.append({'name':name,'circles':1,'tracks':1})
        write_json(self.policy,policy)
        for folder in (self.first,self.second):
            folder.mkdir()
            for i,name in enumerate(sorted(EXPECTED),1):
                (folder/name).write_bytes(f'synthetic official output {i}'.encode())
                (folder/(Path(name).stem+'_rt.dat')).write_bytes(b'timing must not upload')
            write_json(folder/'run_summary.json',{'status':'success','packages':5,'circles':5,'tracks':5,
                                                'wall_seconds':1,'per_package':packets})
            write_json(folder/'logs/client-summary.json',{
                'status':'success','device':'cpu','torch_cuda_initialized':False,
                'counts':{'packages':5,'circles':5,'tracks':5,
                          'per_package':[{'circles':1,'tracks':1} for _ in range(5)]},
                'scans':[{} for _ in range(5)],'handler_max_ms':1,'client_peak_rss_mib':1,
                'package_manifest_sha256':manifest_sha,'versions':{'fixture':True}})
            (folder/'logs/server.log').write_text(''.join(f'[{n}] 圈 1 收到1条航迹(耗时1ms)\n'
                                                        for n in sorted(EXPECTED)),encoding='utf-8')

    def run_build(self):
        return build(self.code,self.inputs,self.first,self.second,self.out,self.report,self.policy)

    def test_upload_is_exactly_five_unchanged_trajectory_files(self):
        result=self.run_build()
        with ZipFile(self.out) as z:
            self.assertEqual(set(z.namelist()),EXPECTED)
            for name in EXPECTED:
                self.assertEqual(z.read(name),(self.first/name).read_bytes())
        self.assertTrue(result['all_trajectories_reproduced_exactly'])
        self.assertFalse(result['website_uploaded'])
        self.assertIsNone(result['official_score'])

    def test_different_repeat_trajectory_is_rejected(self):
        (self.second/'train_003.dat').write_bytes(b'different')
        with self.assertRaisesRegex(ValueError,'different trajectory'):
            self.run_build()
        self.assertFalse(self.out.exists())

    def test_failed_official_run_is_rejected(self):
        path=self.first/'run_summary.json';record=json.loads(path.read_text());record['status']='failed'
        write_json(path,record)
        with self.assertRaisesRegex(ValueError,'Run failed'):
            self.run_build()
        self.assertFalse(self.out.exists())

    def test_mismatched_runtime_code_is_rejected(self):
        path=self.second/'logs/client-summary.json';record=json.loads(path.read_text())
        record['package_manifest_sha256']='0'*64;write_json(path,record)
        with self.assertRaisesRegex(ValueError,'does not match'):
            self.run_build()
        self.assertFalse(self.out.exists())

    def test_wrong_or_extra_input_is_rejected(self):
        (self.inputs/'train_001_rt.dat').write_bytes(b'not an A input')
        with self.assertRaisesRegex(ValueError,'exactly the five'):
            self.run_build()
        self.assertFalse(self.out.exists())

    def test_existing_zip_is_preserved(self):
        self.out.write_bytes(b'preserve this existing file')
        with self.assertRaises(FileExistsError):
            self.run_build()
        self.assertEqual(self.out.read_bytes(),b'preserve this existing file')


if __name__=='__main__':unittest.main()
