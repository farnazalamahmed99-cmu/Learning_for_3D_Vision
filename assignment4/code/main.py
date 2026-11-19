"""Run every step of 16-825 Assignment 4 (Multi-View Transformers).

Usage:
    python main.py <step> [<step> ...]
    python main.py all            # every step below, in order

Each step calls the provided command-line entry points with the settings used for the report.
Steps: part1, lvsm_overfit, lvsm_train, lvsm_eval, lvsm_randS, vggt_overfit, vggt_train,
vggt_eval, vggt_views.
"""

import subprocess
import sys

PY = sys.executable
EVAL_DATA = [
    "--eval-root", "data/scannetpp/prepared_eval", "--num-views", "8",
    "--eval-num-frames", "32", "--image-size", "224", "--seed", "825",
]

STEPS = {
    # Part 1: data visualizations and the final data check.
    "part1": [
        [PY, "scripts/visualize_data.py", "--stage", "unprojection", "--num-scenes", "5", "--max-points", "10000", "--output", "outputs/unprojection"],
        [PY, "scripts/visualize_data.py", "--stage", "canonicalization", "--num-scenes", "5", "--max-points", "10000", "--output", "outputs/canonicalization"],
        [PY, "scripts/check_data.py"],
    ],
    # Part 2.1: memorize one batch.
    "lvsm_overfit": [
        [PY, "-m", "models.lvsm.train", "--overfit", "2000", "--output-dir", "outputs/lvsm_overfit"],
    ],
    # Part 2.2: full training and the source-view / ray-conditioning ablation.
    "lvsm_train": [[PY, "-m", "models.lvsm.train", "--config", "configs/lvsm.json"]],
    "lvsm_eval": [
        [PY, "-m", "models.lvsm.eval", "--checkpoint", "outputs/lvsm/best.pt",
         "--num-source-views", "1", "2", "4", "--conditioning", "correct", "shuffled", "none"],
    ],
    # Part 2.3 (bonus): train with a random number of source views per step, then evaluate.
    "lvsm_randS": [
        [PY, "-m", "models.lvsm.train", "--config", "configs/lvsm_randS.json"],
        [PY, "-m", "models.lvsm.eval", "--checkpoint", "outputs/lvsm_randS/best.pt",
         "--num-source-views", "1", "2", "4", "--conditioning", "correct", "shuffled", "none"],
    ],
    # Part 3.1: single-scene overfit.
    "vggt_overfit": [
        [PY, "-m", "models.vggt.train", "--config", "configs/vggt.json", "--overfit",
         "--output", "outputs/vggt_overfit"],
    ],
    # Part 3.2: full training (4 data workers fit the 16 GB RAM of the AWS g5.xlarge used).
    "vggt_train": [
        [PY, "-m", "models.vggt.train", "--config", "configs/vggt.json",
         "--output", "outputs/vggt_train", "--num-workers", "4"],
    ],
    "vggt_eval": [
        [PY, "-m", "models.vggt.eval", "--checkpoint", "outputs/vggt_train/best.pt", "--split", "val",
         *EVAL_DATA, "--report", "outputs/vggt_train/metrics.json",
         "--visualization", "outputs/vggt_train/visualization/index.html"],
        [PY, "-m", "models.vggt.eval", "--checkpoint", "outputs/vggt_train/best.pt", "--split", "test",
         *EVAL_DATA, "--report", "outputs/vggt_train/test_metrics.json"],
    ],
}

# Part 3.3 (bonus): evaluate the 8-view checkpoint with 2-32 views, then benchmark runtime/memory.
STEPS["vggt_views"] = [
    [PY, "-m", "models.vggt.eval", "--checkpoint", "outputs/vggt_train/best.pt", "--split", split,
     "--eval-root", "data/scannetpp/prepared_eval", "--num-views", str(v), "--eval-num-frames", "32",
     "--image-size", "224", "--seed", "825", "--batch-size", "4",
     "--report", f"outputs/vggt_views/{split}_V{v}.json"]
    for v in (2, 4, 8, 12, 16, 24, 32)
    for split in ("val", "test")
] + [[PY, "extra/bench_views.py", "outputs/vggt_train/best.pt", "outputs/vggt_views/bench.json"]]


def main():
    names = sys.argv[1:] or ["--help"]
    if names == ["all"]:
        names = list(STEPS)
    unknown = [n for n in names if n not in STEPS]
    if unknown:
        print(__doc__)
        if unknown != ["--help"]:
            print("Unknown step(s):", ", ".join(unknown))
        sys.exit(1)
    for name in names:
        for command in STEPS[name]:
            print(f"[{name}] {' '.join(command)}", flush=True)
            subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
