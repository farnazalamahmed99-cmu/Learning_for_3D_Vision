"""
Samples surface points from every training mesh once and saves them, so point-cloud
training does not have to parse the (large) .obj files again for every batch.

Used for Q3.3 (the 3-class dataset, whose car meshes are slow to load):
    python precompute_points.py --out /home/farnaz/data/r2n2_shapenet_dataset_full/points_cache
    python train_model.py --type point --points_cache /home/farnaz/data/r2n2_shapenet_dataset_full/points_cache

Points are sampled exactly like train_model.py does from the loaded mesh
(sample_points_from_meshes on load_obj's verts/faces), just more of them, once.
"""
import argparse
import json
import os
from multiprocessing import Pool

import numpy as np
import torch
from pytorch3d.io import load_obj
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.structures import Meshes

import dataset_location


def sample_model(job):
    synset_id, model_id, out_dir, n_points = job
    out_path = os.path.join(out_dir, synset_id, model_id + ".npy")
    if os.path.exists(out_path):
        return "skipped"
    obj_path = os.path.join(dataset_location.SHAPENET_PATH, synset_id, model_id, "model.obj")
    try:
        verts, faces, _ = load_obj(obj_path, load_textures=False)
        mesh = Meshes(verts=[verts], faces=[faces.verts_idx])
        points = sample_points_from_meshes(mesh, n_points)[0]
    except Exception as e:
        return f"failed {synset_id}/{model_id}: {e}"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.save(out_path, points.numpy().astype(np.float32))
    return "done"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, type=str)
    parser.add_argument("--split", default="train", type=str)
    parser.add_argument("--n_points", default=10000, type=int)
    parser.add_argument("--workers", default=6, type=int)
    args = parser.parse_args()
    torch.set_num_threads(1)

    splits = json.load(open(dataset_location.SPLITS_PATH))[args.split]
    jobs = [(synset_id, model_id, args.out, args.n_points)
            for synset_id, models in splits.items() for model_id in models]
    print(f"{len(jobs)} models to sample")
    with Pool(args.workers) as pool:
        for i, result in enumerate(pool.imap_unordered(sample_model, jobs, chunksize=8)):
            if result.startswith("failed"):
                print(result, flush=True)
            if (i + 1) % 500 == 0:
                print(f"{i + 1}/{len(jobs)}", flush=True)
    print("Done!")
