"""
2.1 Constructs a tetrahedron mesh by hand and renders a 360-degree turntable
gif of it.

Usage:
    python -m starter.render_tetrahedron --output_path images/tetrahedron.gif
"""
import argparse

import imageio
import numpy as np
import pytorch3d
import torch

from starter.utils import get_device, get_mesh_renderer


def get_tetrahedron_mesh(color=[0.7, 0.7, 1], device=None):
    if device is None:
        device = get_device()

    # 4 alternating corners of a cube -> a regular tetrahedron.
    vertices = torch.tensor(
        [
            [1.0, 1.0, 1.0],   # v0
            [-1.0, -1.0, 1.0],  # v1
            [-1.0, 1.0, -1.0],  # v2
            [1.0, -1.0, -1.0],  # v3
        ]
    )

    faces = torch.tensor(
        [
            [0, 1, 3],  # f0
            [0, 3, 2],  # f1
            [0, 2, 1],  # f2
            [3, 1, 2],  # f3
        ]
    )

    textures = torch.ones_like(vertices) * torch.tensor(color)
    mesh = pytorch3d.structures.Meshes(
        verts=vertices.unsqueeze(0),
        faces=faces.unsqueeze(0),
        textures=pytorch3d.renderer.TexturesVertex(textures.unsqueeze(0)),
    )
    return mesh.to(device), vertices.shape[0], faces.shape[0]


def render_tetrahedron_360(
    image_size=256,
    color=[0.7, 0.7, 1],
    num_frames=36,
    dist=5,
    elev=30,
    fps=15,
    output_path="images/tetrahedron.gif",
    device=None,
):
    if device is None:
        device = get_device()

    mesh, num_verts, num_faces = get_tetrahedron_mesh(color, device)
    print(f"tetrahedron: {num_verts} vertices, {num_faces} faces")

    renderer = get_mesh_renderer(image_size=image_size, device=device)
    lights = pytorch3d.renderer.PointLights(location=[[0, 0, -3]], device=device)

    azimuths = torch.linspace(0, 360, num_frames + 1)[:-1]
    images = []
    for azim in azimuths:
        R, T = pytorch3d.renderer.look_at_view_transform(dist=dist, elev=elev, azim=azim.item())
        cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T, device=device)
        rend = renderer(mesh, cameras=cameras, lights=lights)
        image = rend[0, ..., :3].cpu().numpy()
        images.append((image * 255).astype(np.uint8))

    duration = 1000 // fps
    imageio.mimsave(output_path, images, duration=duration, loop=0)
    return num_verts, num_faces


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--num_frames", type=int, default=36)
    parser.add_argument("--output_path", type=str, default="images/tetrahedron.gif")
    args = parser.parse_args()
    render_tetrahedron_360(
        image_size=args.image_size,
        num_frames=args.num_frames,
        output_path=args.output_path,
    )
