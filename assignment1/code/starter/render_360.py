"""
1.1 Renders a 360-degree turntable gif of the cow mesh.

Usage:
    python -m starter.render_360 --num_frames 36 --output_path images/cow_360.gif
"""
import argparse

import imageio
import numpy as np
import pytorch3d
import torch

from starter.utils import get_device, get_mesh_renderer, load_cow_mesh


def render_cow_360(
    cow_path="data/cow.obj",
    image_size=256,
    color=[0.7, 0.7, 1],
    num_frames=36,
    dist=3,
    elev=0,
    fps=15,
    device=None,
):
    if device is None:
        device = get_device()

    # Build the mesh once (same setup as render_mesh.render_cow).
    vertices, faces = load_cow_mesh(cow_path)
    vertices = vertices.unsqueeze(0)  # (N_v, 3) -> (1, N_v, 3)
    faces = faces.unsqueeze(0)  # (N_f, 3) -> (1, N_f, 3)
    textures = torch.ones_like(vertices) * torch.tensor(color)  # (1, N_v, 3)
    mesh = pytorch3d.structures.Meshes(
        verts=vertices,
        faces=faces,
        textures=pytorch3d.renderer.TexturesVertex(textures),
    ).to(device)

    # Renderer and light are also fixed across frames; only the camera moves.
    renderer = get_mesh_renderer(image_size=image_size, device=device)
    lights = pytorch3d.renderer.PointLights(location=[[0, 0, -3]], device=device)

    # Sweep azimuth around a full turn, at a fixed distance and elevation.
    # 360 is excluded since it's the same view as 0.
    azimuths = torch.linspace(0, 360, num_frames + 1)[:-1]

    images = []
    for azim in azimuths:
        R, T = pytorch3d.renderer.look_at_view_transform(dist=dist, elev=elev, azim=azim.item())
        cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T, device=device)
        rend = renderer(mesh, cameras=cameras, lights=lights)
        image = rend[0, ..., :3].cpu().numpy()  # (H, W, 3), floats in [0, 1]
        images.append((image * 255).astype(np.uint8))

    duration = 1000 // fps
    return images, duration


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cow_path", type=str, default="data/cow.obj")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--num_frames", type=int, default=36)
    parser.add_argument("--dist", type=float, default=3)
    parser.add_argument("--elev", type=float, default=0)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--output_path", type=str, default="images/cow_360.gif")
    args = parser.parse_args()

    images, duration = render_cow_360(
        cow_path=args.cow_path,
        image_size=args.image_size,
        num_frames=args.num_frames,
        dist=args.dist,
        elev=args.elev,
        fps=args.fps,
    )
    imageio.mimsave(args.output_path, images, duration=duration, loop=0)
