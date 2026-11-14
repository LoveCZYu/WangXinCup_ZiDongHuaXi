import importlib.util
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from b02.point_tracker import DEFAULT_PROFILE,load_tracker
from b02.replay import INPUT_FIELDS,replay_scene


class ProfileIntegrityTest(unittest.TestCase):
    def test_modified_weight_hash_is_rejected_before_inference(self):
        config=json.loads(DEFAULT_PROFILE.read_text(encoding='utf-8'))
        config['checkpoint']=str((DEFAULT_PROFILE.parent/config['checkpoint']).resolve())
        config['checkpoint_sha256']='0'*64
        with TemporaryDirectory() as folder:
            profile=Path(folder)/'bad.json'
            profile.write_text(json.dumps(config),encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'校验失败'):
                load_tracker(profile,device='cpu')


@unittest.skipUnless(all(importlib.util.find_spec(x) for x in ('torch','numpy','scipy')),'optional point-model dependencies')
class PointPacketTest(unittest.TestCase):
    def test_published_handler_resets_at_packet_boundary_and_preserves_inputs(self):
        handler,_=load_tracker(device='cpu')
        sample=Path(__file__).resolve().parents[1]/'examples/smoke.csv'
        traces=[]
        for _ in range(2):
            trace=io.StringIO()
            def checked(points,state):
                self.assertTrue(all(set(p)==set(INPUT_FIELDS) for p in points))
                return handler(points,state)
            with sample.open(encoding='utf-8') as source:
                summary=replay_scene(source,trace,checked)
            self.assertEqual(summary['circles'],5)
            self.assertEqual(summary['matched_target_points'],4)
            self.assertEqual(summary['clutter_outputs'],0)
            traces.append([json.loads(line)['tracks'] for line in trace.getvalue().splitlines()])
        self.assertEqual(traces[0],traces[1])


if __name__=='__main__':unittest.main()
