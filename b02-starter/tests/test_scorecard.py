import unittest
import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from b02.scorecard import build_scorecard, write_scorecard


def scene(points, hits, outputs, first):
    return {"circles": points, "points": points + 2, "truth_points": points,
            "matched_target_points": hits, "clutter_outputs": 2, "unknown_outputs": 0,
            "per_target": [{"truth_id": 1, "truth_points": points, "matched_points": hits,
                            "output_ids": outputs, "first_output_delay_circles": first,
                            "most_matched_points_with_one_id": max(outputs.values(), default=0)}]}


class ScorecardTest(unittest.TestCase):
    def test_weighted_total_uses_five_subscores_and_exports_same_number(self):
        card = build_scorecard({"scenes": {"a": scene(10, 6, {7: 3, 8: 3}, 7)}}, "test")
        values = card["overall"]["values"]
        self.assertEqual(values["fragmentation_score"], 50)
        self.assertEqual(values["timeliness_score"], 50)
        self.assertAlmostEqual(values["weighted_total"], 53.5)
        with TemporaryDirectory(dir=Path(__file__).parent) as directory:
            root = Path(directory)
            write_scorecard(root, card)
            saved = json.loads((root / "scorecard.json").read_text(encoding="utf-8"))
            with (root / "results.csv").open(encoding="utf-8-sig", newline="") as file:
                rows = list(csv.reader(file))
            self.assertEqual(float(rows[1][1]), saved["overall"]["values"]["weighted_total"])
            self.assertTrue((root / "scorecard-policy.txt").is_file())

    def test_timeliness_limits_missing_short_and_total(self):
        card = build_scorecard({"scenes": {
            "early": {**scene(10, 10, {1: 10}, 0), "clutter_outputs": 0},
            "late": scene(20, 1, {1: 1}, 15),
            "miss": scene(10, 0, {}, None),
            "short": scene(2, 2, {1: 2}, 0),
        }}, "test")["scenes"]
        self.assertEqual(card["early"]["values"]["weighted_total"], 100)
        self.assertEqual(card["late"]["values"]["timeliness_score"], 0)
        self.assertEqual(card["miss"]["values"]["fragmentation_score"], 0)
        self.assertEqual(card["miss"]["values"]["timeliness_score"], 0)
        self.assertIsNone(card["short"]["values"]["weighted_total"])

    def test_track_and_point_averages_and_unmatched_delay(self):
        data = {"scenes": {"a": scene(2, 2, {7: 1, 8: 1}, 0), "b": scene(8, 0, {}, None)}}
        values = build_scorecard(data, "current")["overall"]["values"]
        self.assertEqual(values["point_recall"], 20)
        self.assertEqual(values["track_recall"], 50)
        self.assertEqual(values["dominant_id_coverage"], 25)
        self.assertEqual(values["split_tracks"], 1)
        self.assertEqual(values["delay_median"], 0)
        self.assertEqual(values["unmatched_tracks"], 1)

    def test_baseline_subset_and_undefined_delta(self):
        baseline = {"scenes": {"a": scene(2, 1, {7: 1}, 1), "extra": scene(8, 8, {9: 8}, 0)}}
        current = {"scenes": {"a": scene(2, 0, {}, None)}}
        overall = build_scorecard(current, "current", baseline, "baseline")["overall"]
        self.assertEqual(overall["baseline"]["point_recall"], 50)
        self.assertEqual(overall["delta"]["point_recall"], -50)
        self.assertIsNone(overall["delta"]["delay_median"])
        with self.assertRaisesRegex(ValueError, "无法对照"):
            build_scorecard({"scenes": {"a": scene(3, 1, {7: 1}, 2)}}, "wrong", baseline, "baseline")

    def test_continuous_score_uses_one_id_and_caps_ratio(self):
        data = {"scenes": {"one_id": scene(10, 6, {7: 6}, 2),
                           "split": scene(10, 6, {7: 3, 8: 3}, 2),
                           "early": scene(4, 4, {7: 4}, 0),
                           "short": scene(2, 1, {7: 1}, 1)}}
        card = build_scorecard(data, "current")["scenes"]
        self.assertEqual(card["one_id"]["values"]["continuous_score"], 75)
        self.assertEqual(card["split"]["values"]["continuous_score"], 37.5)
        self.assertEqual(card["early"]["values"]["continuous_score"], 100)
        self.assertIsNone(card["short"]["values"]["continuous_score"])

    def test_clutter_score_uses_known_clutter_and_floors_at_zero(self):
        data = {"scenes": {"known": {**scene(10, 5, {7: 5}, 0), "unknown_outputs": 100},
                           "cluttered": {**scene(4, 4, {7: 4}, 0), "clutter_outputs": 10}}}
        card = build_scorecard(data, "current")["scenes"]
        self.assertEqual(card["known"]["values"]["clutter_score"], 80)
        self.assertEqual(card["cluttered"]["values"]["clutter_score"], 0)
