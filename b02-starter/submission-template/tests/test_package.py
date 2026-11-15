"""Package integrity, packet isolation and official wire-format regression checks."""
import copy
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'src' / 'solution'))
from main import verify_package
from client import encode_tracks
from b02.point_tracker import load_tracker


class PackageTest(unittest.TestCase):
    def test_manifest_and_tampered_weight(self):
        manifest = verify_package()
        self.assertEqual(manifest['team_id'], 'T000906')
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'PACKAGE.json').write_text(json.dumps({'files': {'weights.pt': '0'*64}}))
            (root / 'weights.pt').write_bytes(b'changed weight')
            with self.assertRaisesRegex(ValueError, 'integrity'):
                verify_package(root)

    def test_causal_handler_packet_reset_and_wire_identity(self):
        # Synthetic coordinates only; no training/dev/holdout data in this package.
        handler, metadata = load_tracker(device='cpu')
        self.assertEqual(metadata['output_rule'], {'enter': .525, 'maintain': .05, 'grace': 3})
        sequences = []
        for _ in range(2):
            state, sequence = {}, []
            for scan in range(12):
                points = [dict(Time=5000*scan, CircleID=scan, PntID=10,
                               Azimuth=.3+.0001*scan, Elevation=.25, Distance=.4,
                               Amp=.8, SNR=.7)]
                original = copy.deepcopy(points)
                tracks = handler(points, state)
                self.assertEqual(points, original)
                source = {(p['Time'], p['CircleID'], p['PntID']):
                          (p['Azimuth'], p['Elevation'], p['Distance']) for p in original}
                wire = encode_tracks(tracks, source)
                self.assertEqual(struct.unpack('<I', wire[:4])[0], 1 + 28*len(tracks))
                sequence.append(tracks)
            sequences.append(sequence)
        self.assertTrue(any(sequences[0]), 'Synthetic persistent target should yield output')
        self.assertEqual(sequences[0], sequences[1])


if __name__ == '__main__':
    unittest.main()
