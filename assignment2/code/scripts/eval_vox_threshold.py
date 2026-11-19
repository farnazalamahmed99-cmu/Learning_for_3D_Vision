"""
Q2.4: evaluate the trained voxel model with a different marching-cubes threshold.

Runs eval_model.py unchanged, except that its occupancy level (VOX_THRESHOLD) is replaced,
and the outputs are saved as eval_vox_thr<t>.png and vis/<step>_vox_thr<t>.png (so the
results at the chosen threshold are not overwritten).

Usage (from the assignment2 folder):
    python scripts/eval_vox_threshold.py --threshold 0.3 --load_checkpoint --vis_freq 100 --num_workers 4
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
import mcubes

import eval_model

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Voxel threshold study", parents=[eval_model.get_args_parser()])
    parser.add_argument("--threshold", type=float, required=True)
    args = parser.parse_args()
    args.type = "vox"
    threshold = args.threshold

    # every marching-cubes call (F1 evaluation and rendering) uses the new threshold
    eval_model.VOX_THRESHOLD = threshold
    marching_cubes = mcubes.marching_cubes
    mcubes.marching_cubes = lambda volume, isovalue: marching_cubes(volume, threshold)

    # tag every saved figure with the threshold
    savefig = plt.savefig

    def savefig_tagged(fname, *a, **kw):
        root, ext = os.path.splitext(str(fname))
        return savefig(f"{root}_thr{threshold}{ext or '.png'}", *a, **kw)

    plt.savefig = savefig_tagged
    eval_model.evaluate_model(args)
