import io
import json
from pathlib import Path
import unittest

from b02.algorithm import process_circle
from b02.replay import INPUT_FIELDS, replay_scene


SAMPLE = Path(__file__).resolve().parents[1] / "examples" / "smoke.csv"


class ReplayTest(unittest.TestCase):
    def test_label_isolation_empty_scan_and_identity_retention(self):
        states = []

        def handler(points, state):
            states.append(state)
            for point in points:
                self.assertEqual(set(point), set(INPUT_FIELDS))
            return process_circle(points, state)

        traces = []
        for _ in range(2):
            trace = io.StringIO()
            with SAMPLE.open(encoding="utf-8") as source:
                result = replay_scene(source, trace, handler)
            traces.append([json.loads(line)["tracks"] for line in trace.getvalue().splitlines()])
            self.assertEqual(result["empty_circles"], 1)
            self.assertEqual(result["truth_points"], 8)
            self.assertEqual(result["matched_target_points"], 4)
            self.assertEqual(result["clutter_outputs"], 0)
            self.assertEqual(result["split_truth_tracks"], 0)
            self.assertTrue(all(t["first_output_delay_circles"] == 2 for t in result["per_target"]))

        self.assertEqual(traces[0], traces[1])
        self.assertEqual(traces[0][3], [])
        self.assertIsNot(states[0], states[5])

    def test_modified_input_cannot_pass_original_point_check(self):
        def handler(points, _):
            point = points[0]
            point["Azimuth"] += 0.01
            return [{**point, "BatchID": 1}]

        with SAMPLE.open(encoding="utf-8") as source:
            with self.assertRaisesRegex(ValueError, "原始点"):
                replay_scene(source, io.StringIO(), handler)


if __name__ == "__main__":
    unittest.main()
