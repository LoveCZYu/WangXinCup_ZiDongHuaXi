"""Aggregate bounded diagnostic branches and a scene-separated edge screen."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import rankdata

MODES = ("original", "veto", "restore_motion", "restore_state")
FEATURES = ("affinity", "detector_score", "residual", "rowsize", "track_age",
            "track_gap", "memory_size", "memory_recency", "velocity_innovation", "signal_innovation")


def read(path):
    return json.loads(path.read_bytes())


def outcome(frames, truth):
    counts = Counter()
    for row in frames:
        observed = row["correct_observed"]
        output = row["published_truth"]
        if truth in output:
            counts["correct"] += 1
        elif any(value > 0 for value in output):
            counts["other_target"] += 1
        elif output:
            counts["non_target"] += 1
        else:
            counts["silent"] += 1
        counts["observed"] += observed
        counts["miss_with_target_observed"] += observed and truth not in output
    return counts


def summarize(rows):
    totals = Counter()
    for row in rows:
        totals.update(row["counts"])
    samples = [e for row in rows for e in row["errors"] if "replay" in e]
    paired = [e for e in samples if all(e["replay"][mode]["applied"] for mode in MODES)]
    comparisons = {mode: Counter() for mode in MODES}
    by_scene = defaultdict(lambda: {mode: Counter() for mode in MODES})
    by_case = {}
    for sample in paired:
        expected = sample["expected_truth"]
        key = (sample["scene"], sample["circle"], sample["batch"])
        branch = {}
        for mode in MODES:
            frames = sample["replay"][mode]["frames"]
            assert len(frames)==5
            base = outcome(frames[:1], expected)
            follow = outcome(frames[1:], expected)
            comparisons[mode].update({f"onset_{k}": v for k,v in base.items()})
            comparisons[mode].update({f"follow_{k}": v for k,v in follow.items()})
            comparisons[mode]["has_follow_on_other_target"] += follow["other_target"] > 0
            comparisons[mode]["recoveries"] += follow["correct"] > 0
            by_scene[sample["scene"]][mode].update(follow)
            branch[mode] = dict(follow)
        by_case[key] = branch
    effects = {}
    for mode in MODES[1:]:
        counts = Counter()
        for branches in by_case.values():
            old, new = branches["original"], branches[mode]
            change = new.get("other_target", 0)-old.get("other_target", 0)
            counts["fewer_wrong" if change<0 else "more_wrong" if change>0 else "same_wrong"] += 1
            counts["wrong_to_silent_cases"] += change<0 and new.get("silent",0)>old.get("silent",0)
        effects[mode] = dict(counts)
    return dict(totals=dict(totals), replayed=len(samples), paired=len(paired),
                correct_edge_available=sum(e["correct_edge_available"] for e in paired),
                wrong_written_to_memory=sum(e["trusted_memory_update"] for e in paired),
                comparisons={k: dict(v) for k,v in comparisons.items()}, effects=effects,
                scenes={scene:{mode:dict(counts) for mode,counts in modes.items()}
                        for scene,modes in by_scene.items()})


def features(sample):
    track, edge = sample["track_features"], sample["edge_features"]
    return [sample["affinity"], sample["detector_score"], edge[9], len_hint(sample),
            track[35], track[38], track[93], track[94],
            float(np.linalg.norm(edge[6:9])), float(np.linalg.norm(edge[16:18]))]


def len_hint(sample):
    return float(np.log1p(sample["candidate_count"]))


def cohort(rows):
    return [(row["scene"], features(edge), label) for row in rows
            for label, name in ((1, "errors"), (0, "correct_controls"))
            for edge in row[name] if edge["strong_output"]]


def fit(data):
    x = np.array([record[1] for record in data])
    y = np.array([record[2] for record in data])
    if len(set(y)) != 2:
        raise ValueError("Each fit fold requires both error and correct controls")
    mean, scale = x.mean(0), np.maximum(x.std(0), .01)
    design = np.column_stack(((x-mean)/scale, np.ones(len(x))))
    weights = np.where(y == 1, .5/y.sum(), .5/(1-y).sum())
    def objective(coef):
        z = design @ coef
        penalty = np.r_[coef[:-1], 0.]
        loss = np.dot(weights, np.logaddexp(0., z)-y*z)+.05*np.dot(penalty, penalty)
        grad = design.T @ (weights*(expit(z)-y))+.10*penalty
        return loss, grad
    result = minimize(objective, np.zeros(design.shape[1]), jac=True, method="L-BFGS-B")
    if not result.success:
        raise ValueError(result.message)
    return mean, scale, result.x


def predict(model, data):
    mean, scale, coef = model
    x = np.array([record[1] for record in data])
    return ((x-mean)/scale) @ coef[:-1]+coef[-1]


def metric(scores, data, threshold, old_affinity=False):
    y = np.array([r[2] for r in data], dtype=bool)
    positive, negative = int(y.sum()), int((~y).sum())
    exposed = scores >= threshold
    auc = ((rankdata(scores)[y].sum()-positive*(positive+1)/2)/(positive*negative)) if positive and negative else None
    return dict(errors=positive, controls=negative, flagged_errors=int((y & exposed).sum()),
                flagged_controls=int((~y & exposed).sum()), auc=auc, old_affinity=old_affinity)


def screen(development, work):
    train, evaluation = cohort(development), cohort(work)
    if sum(row[2] for row in train) < 20 or len({row[0] for row in train if row[2]}) < 4:
        return dict(status="insufficient development errors", development_counts=len(train), work_counts=len(evaluation))
    held_out, decisions, reference_decisions, ordered = [], [], [], []
    for scene in sorted({row[0] for row in train}):
        fit_rows = [row for row in train if row[0]!=scene]
        test_rows = [row for row in train if row[0]==scene]
        model = fit(fit_rows)
        held_out.extend(predict(model, test_rows))
        controls = [row for row in fit_rows if not row[2]]
        threshold = float(np.quantile(predict(model, controls), .99, method="higher"))
        ref_cut = float(np.quantile([-row[1][0] for row in controls], .99, method="higher"))
        decisions.extend(predict(model, test_rows) > threshold)
        reference_decisions.extend(-row[1][0] > ref_cut for row in test_rows)
        ordered.extend(test_rows)
    model = fit(train)
    train_controls = [row for row in train if not row[2]]
    threshold = float(np.quantile(predict(model, train_controls), .99, method="higher"))
    ref_cut = float(np.quantile([-row[1][0] for row in train_controls], .99, method="higher"))
    loso = metric(np.array(held_out), ordered, float("inf"))
    loso["flagged_errors"] = sum(bool(flag) and bool(row[2]) for flag,row in zip(decisions,ordered))
    loso["flagged_controls"] = sum(bool(flag) and not row[2] for flag,row in zip(decisions,ordered))
    loso["old_affinity_flagged_errors"] = sum(bool(flag) and bool(row[2]) for flag,row in zip(reference_decisions,ordered))
    loso["old_affinity_flagged_controls"] = sum(bool(flag) and not row[2] for flag,row in zip(reference_decisions,ordered))
    return dict(status="diagnostic_only", feature_names=list(FEATURES), development_leave_one_scene_out=loso,
                work=metric(predict(model,evaluation),evaluation,np.nextafter(threshold,np.inf)),
                work_old_affinity=metric(np.array([-row[1][0] for row in evaluation]),evaluation,
                                         np.nextafter(ref_cut,np.inf),old_affinity=True),
                threshold=threshold, threshold_selection="99th percentile of fit controls, strict above",
                scope="Enriched sampled rows, flag rates do not estimate deployment precision.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    import hashlib
    raw = args.protocol.read_bytes()
    plan = json.loads(raw)
    digest = hashlib.sha256(raw).hexdigest()
    rows = [read(path) for path in args.data.glob("propagation-*/*.json")]
    by_role = defaultdict(list)
    for row in rows:
        assert row["protocol_sha256"]==digest
        by_role[row["role"]].append(row)
        for edge in row["errors"]:
            edge["scene"] = row["scene"]
    for role, group in by_role.items():
        assert len(group)==len({row["scene"] for row in group}) and {row["scene"] for row in group}==set(plan[role]["scenes"])
    assert set(by_role)=={"development", "work"}
    output = dict(protocol_sha256=digest, groups={name:summarize(group) for name,group in by_role.items()},
                  screen=screen(by_role["development"],by_role["work"]))
    args.out.write_text(json.dumps(output,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(output,indent=2))


if __name__ == "__main__":
    main()
