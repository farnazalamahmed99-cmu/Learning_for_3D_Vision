"""
Q2.1: renders chosen test chairs (input RGB | predicted voxels | ground-truth mesh) for the
trained voxel model at eval_model.VOX_THRESHOLD, with each chair's own F1@0.05 in the
title. Reuses the helpers in eval_model.py and does not touch the official eval outputs.

Usage (from the assignment2 folder):
    python scripts/render_extra_chairs.py 75 175 275 350
Images are saved to vis_extra/<chair>_vox.png.
"""
import argparse
import contextlib
import io
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from pytorch3d.datasets.r2n2.utils import collate_batched_R2N2

import dataset_location
import eval_model as E
from model import SingleViewto3D
from r2n2_custom import R2N2
from render_utils import voxels_to_mesh

if __name__ == "__main__":
    chair_ids = [int(i) for i in sys.argv[1:]]
    args = argparse.Namespace(arch="resnet18", type="vox", device="cuda", load_feat=False,
                              batch_size=1, n_points=1000)
    thresholds = [0.01, 0.02, 0.03, 0.04, 0.05]
    torch.manual_seed(0)
    with contextlib.redirect_stdout(io.StringIO()):
        dataset = R2N2("test", dataset_location.SHAPENET_PATH, dataset_location.R2N2_PATH,
                       dataset_location.SPLITS_PATH, return_voxels=True)
    model = SingleViewto3D(args).to(args.device)
    checkpoint = torch.load("checkpoint_vox.pth", map_location=args.device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    os.makedirs("vis_extra", exist_ok=True)
    for idx in chair_ids:
        images, mesh_gt = E.preprocess(collate_batched_R2N2([dataset[idx]]), args)
        with torch.no_grad():
            pred = model(images, args)
        f1 = E.evaluate(pred, mesh_gt, thresholds, args)["F1@0.050000"].item()
        gt_vis = E.color_mesh(mesh_gt, [1, 0.7, 0.7], args.device)
        if (pred[0, 0] > E.VOX_THRESHOLD).any():
            pred_mesh = voxels_to_mesh(pred[0, 0], device=args.device, isovalue=E.VOX_THRESHOLD)
            pred_img = E.render_mesh(E.align_vox_mesh(pred_mesh, gt_vis), args.device)
        else:
            pred_img = np.ones((256, 256, 3))
        panels = [images[0].cpu().numpy(), pred_img, E.render_mesh(gt_vis, args.device)]
        titles = ["Input RGB", f"Predicted voxels (F1@0.05 = {f1:.1f})", "Ground truth mesh"]
        fig, axes = plt.subplots(1, 3, figsize=(9, 3))
        for ax, img, title in zip(axes, panels, titles):
            ax.imshow(img)
            ax.set_title(title, fontsize=10)
            ax.axis("off")
        plt.savefig(f"vis_extra/{idx}_vox.png", bbox_inches="tight")
        plt.close(fig)
        print(f"chair {idx}: F1@0.05 = {f1:.1f}")
