"""
(Extra Credit) 7. Sampling Points on Meshes: sample a point cloud from a
triangle mesh via area-weighted stratified sampling, and compare against the
original mesh at several sample counts.

Usage:
    python -m starter.sample_mesh --output_path images/sampled_cow.gif
"""
import argparse

import imageio
import numpy as np
import pytorch3d
import torch

from starter.utils import get_device, get_mesh_renderer, get_points_renderer, load_cow_mesh


def sample_points_from_mesh(vertices, faces, num_samples):
    """
    Uniformly samples `num_samples` points from the surface of a triangle mesh
    using area-weighted stratified sampling:
        1. Sample a face with probability proportional to its area.
        2. Sample a random barycentric coordinate uniformly.
        3. Compute the corresponding surface point.

    Args:
        vertices (torch.Tensor): (N_v, 3) mesh vertices.
        faces (torch.Tensor): (N_f, 3) triangle vertex indices.
        num_samples (int): Number of points to sample.

    Returns:
        points (torch.Tensor): (num_samples, 3) sampled points.
    """
    tris = vertices[faces]  # (N_f, 3, 3)
    v0, v1, v2 = tris[:, 0], tris[:, 1], tris[:, 2]
    face_areas = 0.5 * torch.linalg.cross(v1 - v0, v2 - v0).norm(dim=1)  # (N_f,)

    face_idx = torch.multinomial(face_areas, num_samples, replacement=True)
    v0, v1, v2 = v0[face_idx], v1[face_idx], v2[face_idx]

    # Uniform random barycentric coordinates: sample 2 uniform variables and
    # fold the ones that fall outside the triangle back in, so the
    # distribution is uniform over the triangle (not just the parallelogram).
    u = torch.rand(num_samples)
    v = torch.rand(num_samples)
    outside = (u + v) > 1
    u[outside] = 1 - u[outside]
    v[outside] = 1 - v[outside]
    w = 1 - u - v

    points = w.unsqueeze(1) * v0 + u.unsqueeze(1) * v1 + v.unsqueeze(1) * v2
    return points


def render_sampled_cow(
    cow_path="data/cow.obj",
    image_size=256,
    sample_counts=(10, 100, 1000, 10000),
    num_frames=36,
    dist=3,
    output_path="images/sampled_cow.gif",
    device=None,
):
    """
    Renders the original cow mesh next to point clouds sampled from it at
    several sample counts, side by side, as a 360-degree turntable gif.
    """
    if device is None:
        device = get_device()
    vertices, faces = load_cow_mesh(cow_path)

    mesh_renderer = get_mesh_renderer(image_size=image_size, device=device)
    points_renderer = get_points_renderer(image_size=image_size, device=device, radius=0.01)
    mesh_lights = pytorch3d.renderer.PointLights(location=[[0, 0, -3]], device=device)

    textures = torch.ones_like(vertices) * torch.tensor([0.7, 0.7, 1.0])
    cow_mesh = pytorch3d.structures.Meshes(
        verts=vertices.unsqueeze(0),
        faces=faces.unsqueeze(0),
        textures=pytorch3d.renderer.TexturesVertex(textures.unsqueeze(0)),
    ).to(device)

    renderables = [("mesh", cow_mesh)]
    for n in sample_counts:
        points = sample_points_from_mesh(vertices, faces, n).to(device)
        color = torch.ones_like(points) * torch.tensor([0.7, 0.7, 1.0], device=device)
        point_cloud = pytorch3d.structures.Pointclouds(
            points=points.unsqueeze(0), features=color.unsqueeze(0)
        )
        renderables.append((f"{n}_points", point_cloud))

    azimuths = torch.linspace(0, 360, num_frames + 1)[:-1]
    frames = []
    for azim in azimuths:
        R, T = pytorch3d.renderer.look_at_view_transform(dist=dist, elev=0, azim=azim.item())
        cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T, device=device)
        panels = []
        for name, renderable in renderables:
            if name == "mesh":
                rend = mesh_renderer(renderable, cameras=cameras, lights=mesh_lights)
            else:
                rend = points_renderer(renderable, cameras=cameras)
            panels.append(rend[0, ..., :3].detach().cpu().numpy().clip(0, 1))
        frame = (np.concatenate(panels, axis=1) * 255).astype(np.uint8)
        frames.append(frame)

    imageio.mimsave(output_path, frames, duration=1000 // 15, loop=0)
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cow_path", type=str, default="data/cow.obj")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--num_frames", type=int, default=36)
    parser.add_argument("--output_path", type=str, default="images/sampled_cow.gif")
    args = parser.parse_args()
    render_sampled_cow(
        cow_path=args.cow_path,
        image_size=args.image_size,
        num_frames=args.num_frames,
        output_path=args.output_path,
    )
