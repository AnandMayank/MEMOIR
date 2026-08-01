"""
Independent-judge check for the 70B/gpt-4o-mini MEMOIR ucb+consolidate results
(eval_ucb_consolidate_backbone.py). Both those runs self-judge: the same
backbone that generates the answer also judges its own correctness, which can
bias the accuracy number independent of true answer quality.

Re-judges every (question, gold, predicted) triple from both per-QA dumps
using a single FIXED, independent judge (llama3.1:8b via Ollama, local, free,
not involved in generating either set of answers) with the same judge prompt
already used throughout this project (eval_qa_together.judge_prompt), so any
accuracy shift isolates the self-judging effect from a real quality change.

Usage:
  python benchmarks/memoir/independent_judge_check.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from ollama_client import OllamaClient
from eval_qa_together import judge_prompt


def rejudge(path: Path, judge: OllamaClient) -> dict:
    rows = [json.loads(l) for l in open(path)]
    n_flip_to_correct = 0
    n_flip_to_wrong = 0
    rejudged_correct = 0
    out_rows = []
    for r in rows:
        text, _lat = judge._call(
            judge_prompt(r["question"], r["gold"], r["predicted"]), max_tokens=4)
        new_correct = text.strip().lower().startswith("yes")
        old_correct = r["acc"] == 1.0
        if new_correct and not old_correct:
            n_flip_to_correct += 1
        elif not new_correct and old_correct:
            n_flip_to_wrong += 1
        rejudged_correct += int(new_correct)
        out_rows.append({**r, "acc_independent_judge": 1.0 if new_correct else 0.0})
    n = len(rows)
    return {
        "n": n,
        "acc_self_judged": sum(r["acc"] for r in rows) / n,
        "acc_independent_judge": rejudged_correct / n,
        "flips_self_wrong_to_independent_correct": n_flip_to_correct,
        "flips_self_correct_to_independent_wrong": n_flip_to_wrong,
        "rows": out_rows,
    }


def main() -> None:
    judge = OllamaClient(model="llama3.1:8b")
    for name, path in [
        ("70b", _REPO / "results" / "ucb_consolidate_70b_perqa.jsonl"),
        ("gpt4omini", _REPO / "results" / "ucb_consolidate_gpt4omini_perqa.jsonl"),
    ]:
        print(f"=== {name} ===", flush=True)
        result = rejudge(path, judge)
        print(f"  n={result['n']}")
        print(f"  acc (self-judged):        {result['acc_self_judged']:.4f}")
        print(f"  acc (independent judge):  {result['acc_independent_judge']:.4f}")
        print(f"  flips wrong->correct: {result['flips_self_wrong_to_independent_correct']}, "
              f"correct->wrong: {result['flips_self_correct_to_independent_wrong']}", flush=True)
        out_path = _REPO / "results" / f"ucb_consolidate_{name}_independent_judge.json"
        with open(out_path, "w") as f:
            json.dump({k: v for k, v in result.items() if k != "rows"}, f, indent=2)
        perqa_path = _REPO / "results" / "perqa" / f"ucb_consolidate_{name}_independent_judge.jsonl"
        perqa_path.parent.mkdir(parents=True, exist_ok=True)
        with open(perqa_path, "w") as f:
            for row in result["rows"]:
                f.write(json.dumps(row) + "\n")
        print(f"  Saved to {out_path}", flush=True)


if __name__ == "__main__":
    main()
