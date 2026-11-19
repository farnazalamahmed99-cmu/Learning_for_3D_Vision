import argparse
import os
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import pytorch3d.renderer
from pytorch3d.ops import knn_points, sample_points_from_meshes
from pytorch3d.structures import Meshes, Pointclouds

import dataset_location
from model import SingleViewto3D
from r2n2_custom import R2N2
from render_utils import get_mesh_renderer, get_points_renderer

DEVICE = "cuda"
ERR_MAX = 0.05  # errors at or above the F1 threshold are drawn fully red
def load_model(model_type):
    args = argparse.Namespace(arch="resnet18", type=model_type, device=DEVICE, load_feat=False,
                              n_points=1000, batch_size=1)
    model = SingleViewto3D(args).to(DEVICE)
    checkpoint = torch.load(f"checkpoint_{model_type}.pth", map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, args


def load_chair(dataset, idx):
    
    item = dataset[idx]
    image = item["images"].unsqueeze(0).to(DEVICE)
    mesh_gt = Meshes(verts=[item["verts"].to(DEVICE)], faces=[item["faces"].to(DEVICE)])
    return image, mesh_gt
def get_camera(azim=45, elev=20, dist=1.5):
    R, T = pytorch3d.renderer.look_at_view_transform(dist=dist, elev=elev, azim=azim)
    return pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T, device=DEVICE)
def render_colored_points(points, colors):
    # points: n x 3, colors: n x 3 in [0, 1]
    rgba = torch.cat([colors, torch.ones_like(colors[:, :1])], dim=1)
    cloud = Pointclouds(points=[points], features=[rgba])
    renderer = get_points_renderer(image_size=256, radius=0.015, device=DEVICE)
    return renderer(cloud, cameras=get_camera())[0, ..., :3].cpu().numpy().clip(0, 1)
def render_colored_mesh(verts, faces, colors):
    mesh = Meshes(verts=[verts], faces=[faces],
                  textures=pytorch3d.renderer.TexturesVertex(verts_features=[colors]))
    renderer = get_mesh_renderer(image_size=256, device=DEVICE)
    lights = pytorch3d.renderer.PointLights(location=[[0, 0, -3]], device=DEVICE)
    return renderer(mesh, cameras=get_camera(), lights=lights)[0, ..., :3].cpu().numpy().clip(0, 1)

def nearest_dist(a, b):
    # for every point in a (n x 3): distance to the closest point in b (m x 3)
    return knn_points(a[None], b[None], K=1).dists[0, :, 0].sqrt()


def error_colors(dists):
    # blue = accurate, red = at least ERR_MAX away
    cmap = plt.get_cmap("jet")
    rgb = cmap((dists / ERR_MAX).clamp(0, 1).cpu().numpy())[:, :3]
    return torch.tensor(rgb, dtype=torch.float32, device=DEVICE)
def part1_error_heatmaps(dataset, point_model, point_args, mesh_model, mesh_args, chair_ids):
    for idx in chair_ids:
        image, mesh_gt = load_chair(dataset, idx)
        gt_dense = sample_points_from_meshes(mesh_gt, 20000)[0]   # stands in for the true surface
        gt_show = gt_dense[:5000]                                  # fewer points for drawing
        with torch.no_grad():
            points = point_model(image, point_args)[0]            # 1000 x 3
            mesh_pred = mesh_model(image, mesh_args)
        mesh_verts = mesh_pred.verts_packed()
        mesh_faces = mesh_pred.faces_packed()
        mesh_samples = sample_points_from_meshes(mesh_pred, 20000)[0]

        panels = [
            (image[0].cpu().numpy(), "Input RGB"),
            (render_colored_points(points, error_colors(nearest_dist(points, gt_dense))),
             "Point: prediction error"),
            (render_colored_points(gt_show, error_colors(nearest_dist(gt_show, points))),
             "Point: missed GT parts"),
            (render_colored_mesh(mesh_verts, mesh_faces, error_colors(nearest_dist(mesh_verts, gt_dense))),
             "Mesh: prediction error"),
            (render_colored_points(gt_show, error_colors(nearest_dist(gt_show, mesh_samples))),
             "Mesh: missed GT parts"),
        ]
        fig, axes = plt.subplots(1, 5, figsize=(17, 3.6))
        for ax, (img, title) in zip(axes, panels):
            ax.imshow(img)
            ax.set_title(title, fontsize=10)
            ax.axis("off")
        colors = plt.cm.ScalarMappable(cmap="jet", norm=plt.Normalize(0, ERR_MAX))
        fig.colorbar(colors, ax=axes, fraction=0.015, label="distance to nearest point")
        plt.savefig(f"interpret/errors_{idx}.png", bbox_inches="tight")
        plt.close(fig)
def part1_error_by_height(dataset, point_model, point_args, mesh_model, mesh_args, n_chairs=200, n_bins=10):
    sums = {"point": np.zeros(n_bins), "mesh": np.zeros(n_bins)}
    counts = np.zeros(n_bins)
    step = max(1, len(dataset) // n_chairs)
    for idx in range(0, len(dataset), step)[:n_chairs]:
        image, mesh_gt = load_chair(dataset, idx)
        gt = sample_points_from_meshes(mesh_gt, 5000)[0]
        with torch.no_grad():
            points = point_model(image, point_args)[0]
            mesh_samples = sample_points_from_meshes(mesh_model(image, mesh_args), 5000)[0]
        # height of each GT point inside its chair: 0 = floor, 1 = top of the back
        y = gt[:, 1]
        height = ((y - y.min()) / (y.max() - y.min())).cpu().numpy()
        bins = np.minimum((height * n_bins).astype(int), n_bins - 1)
        counts += np.bincount(bins, minlength=n_bins)
        for name, pred in (("point", points), ("mesh", mesh_samples)):
            d = nearest_dist(gt, pred).cpu().numpy()
            sums[name] += np.bincount(bins, weights=d, minlength=n_bins)

    centers = (np.arange(n_bins) + 0.5) / n_bins
    plt.figure(figsize=(5, 4))
    for name in sums:
        plt.plot(sums[name] / counts, centers, marker="o", label=name)
    plt.xlabel("mean distance from GT surface to nearest prediction")
    plt.ylabel("height in chair (0 = floor, 1 = top)")
    plt.title("Where reconstructions miss")
    plt.legend()
    plt.savefig("interpret/error_by_height.png", bbox_inches="tight")
    plt.close()
def part2_index_colors(dataset, point_model, point_args, chair_ids):
    predictions, images = [], []
    for idx in chair_ids:
        image, _ = load_chair(dataset, idx)
        with torch.no_grad():
            predictions.append(point_model(image, point_args)[0])
        images.append(image[0].cpu().numpy())

    # give every output index a colour from where it lands in the first chair (x, y, z -> r, g, b)
    ref = predictions[0]
    colors = (ref - ref.min(0).values) / (ref.max(0).values - ref.min(0).values)

    fig, axes = plt.subplots(2, len(chair_ids), figsize=(3.2 * len(chair_ids), 6.4))
    for col, (img, points) in enumerate(zip(images, predictions)):
        axes[0, col].imshow(img)
        axes[0, col].set_title(f"test chair {chair_ids[col]}")
        axes[0, col].axis("off")
        axes[1, col].imshow(render_colored_points(points, colors))
        axes[1, col].axis("off")
    axes[1, 0].set_title("colour = output index\n(set by the first chair)", fontsize=9)
    plt.savefig("interpret/index_colors.png", bbox_inches="tight")
    plt.close(fig)
def part2_index_consistency(dataset, point_model, point_args, n_chairs=200):
    clouds = []
    step = max(1, len(dataset) // n_chairs)
    for idx in range(0, len(dataset), step)[:n_chairs]:
        image, _ = load_chair(dataset, idx)
        with torch.no_grad():
            p = point_model(image, point_args)[0]
        # scale each chair into the unit box so chair size does not matter
        clouds.append((p - p.min(0).values) / (p.max(0).values - p.min(0).values))
    clouds = torch.stack(clouds)                                   # chairs x 1000 x 3
    same_index = clouds.std(dim=0).mean().item()                   # how far one output index moves between chairs
    shuffled = torch.stack([c[torch.randperm(c.shape[0], device=c.device)] for c in clouds])
    random_index = shuffled.std(dim=0).mean().item()               # same, if indices had no meaning
    print(f"spread of one output index across chairs: {same_index:.3f}")
    print(f"spread with shuffled indices (no correspondence): {random_index:.3f}")
if __name__ == "__main__":
    random.seed(0)          # the dataset picks a random view per chair: make it repeatable
    torch.manual_seed(0)
    os.makedirs("interpret", exist_ok=True)

    dataset = R2N2("test", dataset_location.SHAPENET_PATH, dataset_location.R2N2_PATH,
                   dataset_location.SPLITS_PATH, return_voxels=False)
    point_model, point_args = load_model("point")
    mesh_model, mesh_args = load_model("mesh")

    # part 1: where do the models fail?
    part1_error_heatmaps(dataset, point_model, point_args, mesh_model, mesh_args, chair_ids=[0, 100, 275, 500])
    part1_error_by_height(dataset, point_model, point_args, mesh_model, mesh_args)

    # part 2: does each output index of the point model stick to one part?
    part2_index_colors(dataset, point_model, point_args, chair_ids=[275, 0, 100, 500, 75])
    part2_index_consistency(dataset, point_model, point_args)

