"""Runs every experiment in Assignment 5 with the settings used for the website.

    python main.py all                     # everything, in order
    python main.py q1 q2                   # chosen steps only
    python main.py q3_points q3_rot q4     # (q3 uses the q1/q2 checkpoints)
"""
import subprocess
import sys

SEG_OBJECTS = [0, 471, 434, 351, 577]          # chairs shown on the website (351 and 577 are the bad cases)
NUM_POINTS = [10000, 5000, 1000, 500, 100, 50, 10]
ROT_DEGS = [0, 15, 30, 45, 60, 90, 180]
DGCNN = ["--model", "dgcnn", "--num_points", "1024", "--checkpoint_dir", "./checkpoints_dgcnn"]


def run(*args):
    cmd = [sys.executable, *map(str, args)]
    print("\n>>", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def q1():
    run("train.py", "--task", "cls", "--num_epochs", 50)
    run("eval_cls.py", "--viz", "--exp_name", "q1")


def q2():
    run("train.py", "--task", "seg", "--num_epochs", 30)
    for i in SEG_OBJECTS:
        run("eval_seg.py", "--i", i, "--exp_name", f"obj{i}")


def q3_points():
    for n in NUM_POINTS:
        run("eval_cls.py", "--num_points", n, "--viz", "--exp_name", f"q3_pts{n}")
        run("eval_seg.py", "--num_points", n, "--i", 0, "--exp_name", f"q3_pts{n}")


def q3_rot():
    for d in ROT_DEGS:
        run("eval_cls.py", "--rot_deg", d, "--viz", "--exp_name", f"q3_rot{d}")
        run("eval_seg.py", "--rot_deg", d, "--i", 0, "--exp_name", f"q3_rot{d}")


def q4():
    run("train.py", "--task", "cls", "--num_epochs", 40, "--exp_name", "dgcnn", *DGCNN)
    run("train.py", "--task", "seg", "--num_epochs", 50, "--exp_name", "dgcnn", *DGCNN)
    run("eval_cls.py", "--viz", "--exp_name", "q4dgcnn", *DGCNN)
    for i in SEG_OBJECTS:
        run("eval_seg.py", "--i", i, "--exp_name", f"q4dgcnn_{i}", *DGCNN)
        run("eval_seg.py", "--num_points", 1024, "--i", i, "--exp_name", f"q4pointnet_{i}")


STEPS = {"q1": q1, "q2": q2, "q3_points": q3_points, "q3_rot": q3_rot, "q4": q4}

if __name__ == "__main__":
    names = sys.argv[1:] or ["all"]
    if names == ["all"]:
        names = list(STEPS)
    bad = [n for n in names if n not in STEPS]
    if bad:
        sys.exit(f"unknown step(s) {bad}; choose from {list(STEPS)} or 'all'")
    for n in names:
        STEPS[n]()
