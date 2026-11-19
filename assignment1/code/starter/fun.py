"""
(Extra Credit) 6. Do Something Fun: builds a Menger-sponge-style fractal out
of many small cube meshes (reusing the cube from Part 2.2) and renders a
360-degree turntable gif.

Usage:
    python -m starter.fun --level 2 --output_path images/fun_menger_sponge.gif
"""
import argparse

import imageio
import numpy as np
import pytorch3d
import torch

from starter.utils import get_device, get_mesh_renderer


def _menger_centers(level):
    """
    Recursively computes the centers (in a [-1, 1]^3 cube) and half-extent of
    every small cube that survives `level` rounds of Menger-sponge removal
    (remove the center of each face and the very center, keeping the 20
    "corner + edge" sub-cubes at each level).
    """
    centers = [torch.zeros(3)]
    scale = 1.0
    for _ in range(level):
        scale /= 3.0
        new_centers = []
        offsets = [-2 * scale, 0.0, 2 * scale]
        for c in centers:
            for ox in offsets:
                for oy in offsets:
                    for oz in offsets:
                        # Keep the sub-cube unless it is a face-center or the
                        # volume-center (i.e. at least 2 of the 3 offsets must
                        # be non-zero).
                        num_zero = (ox == 0) + (oy == 0) + (oz == 0)
                        if num_zero >= 2:
                            continue
                        new_centers.append(c + torch.tensor([ox, oy, oz]))
        centers = new_centers
    return centers, scale


def get_menger_sponge_mesh(level=2, color=[0.9, 0.5, 0.2], device=None):
    """
    Builds a Menger-sponge fractal by instancing many small cubes (same
    vertex/face layout as get_cube_mesh in render_cube.py).
    """
    if device is None:
        device = get_device()

    base_vertices = torch.tensor(
        [
            [-1.0, -1.0, -1.0],
            [1.0, -1.0, -1.0],
            [1.0, 1.0, -1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, -1.0, 1.0],
            [1.0, -1.0, 1.0],
            [1.0, 1.0, 1.0],
            [-1.0, 1.0, 1.0],
        ]
    )
    base_faces = torch.tensor(
        [
            [0, 2, 1], [0, 3, 2],
            [4, 5, 6], [4, 6, 7],
            [0, 7, 3], [0, 4, 7],
            [1, 6, 5], [1, 2, 6],
            [0, 5, 4], [0, 1, 5],
            [3, 6, 2], [3, 7, 6],
        ]
    )

    centers, scale = _menger_centers(level)
    all_verts, all_faces = [], []
    for center in centers:
        verts = base_vertices * scale + center
        all_verts.append(verts)
        all_faces.append(base_faces)

    verts = torch.stack(all_verts)  # (num_cubes, 8, 3)
    faces = torch.stack(all_faces)  # (num_cubes, 12, 3)
    offsets = (torch.arange(len(centers)) * 8).view(-1, 1, 1)
    faces = faces + offsets

    verts = verts.reshape(1, -1, 3)
    faces = faces.reshape(1, -1, 3)
    textures = torch.ones_like(verts) * torch.tensor(color)
    mesh = pytorch3d.structures.Meshes(
        verts=verts, faces=faces, textures=pytorch3d.renderer.TexturesVertex(textures)
    )
    return mesh.to(device), len(centers)


def render_menger_sponge(
    level=2,
    image_size=256,
    num_frames=36,
    dist=6,
    elev=25,
    output_path="images/fun_menger_sponge.gif",
    device=None,
):
    if device is None:
        device = get_device()
    mesh, num_cubes = get_menger_sponge_mesh(level=level, device=device)
    print(
        f"Menger sponge level {level}: {num_cubes} cubes, "
        f"{mesh.verts_packed().shape[0]} vertices, {mesh.faces_packed().shape[0]} faces"
    )

    renderer = get_mesh_renderer(image_size=image_size, device=device)
    lights = pytorch3d.renderer.PointLights(location=[[2, 2, -4]], device=device)

    azimuths = torch.linspace(0, 360, num_frames + 1)[:-1]
    images = []
    for azim in azimuths:
        R, T = pytorch3d.renderer.look_at_view_transform(dist=dist, elev=elev, azim=azim.item())
        cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T, device=device)
        rend = renderer(mesh, cameras=cameras, lights=lights)
        image = rend[0, ..., :3].cpu().numpy()
        images.append((image * 255).astype(np.uint8))

    imageio.mimsave(output_path, images, duration=1000 // 15, loop=0)
    return num_cubes


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", type=int, default=2)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--num_frames", type=int, default=36)
    parser.add_argument("--output_path", type=str, default="images/fun_menger_sponge.gif")
    args = parser.parse_args()
    render_menger_sponge(
        level=args.level,
        image_size=args.image_size,
        num_frames=args.num_frames,
        output_path=args.output_path,
    )
