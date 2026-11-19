"""
Sample code to render various representations.

Usage:
    python -m starter.render_generic --render point_cloud  # 5.1
    python -m starter.render_generic --render parametric  --num_samples 100  # 5.2
    python -m starter.render_generic --render implicit  # 5.3
"""
import argparse
import pickle

import imageio
import matplotlib.pyplot as plt
import mcubes
import numpy as np
import pytorch3d
import torch

from starter.utils import (
    get_device,
    get_mesh_renderer,
    get_points_renderer,
    render_turntable_gif,
    unproject_depth_image,
)


def load_rgbd_data(path="data/rgbd_data.pkl"):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data


def render_bridge(
    point_cloud_path="data/bridge_pointcloud.npz",
    image_size=256,
    background_color=(1, 1, 1),
    device=None,
):
    """
    Renders a point cloud.
    """
    if device is None:
        device = get_device()
    renderer = get_points_renderer(
        image_size=image_size, background_color=background_color
    )
    point_cloud = np.load(point_cloud_path)
    verts = torch.Tensor(point_cloud["verts"][::50]).to(device).unsqueeze(0)
    rgb = torch.Tensor(point_cloud["rgb"][::50]).to(device).unsqueeze(0)
    point_cloud = pytorch3d.structures.Pointclouds(points=verts, features=rgb)
    R, T = pytorch3d.renderer.look_at_view_transform(4, 10, 0)
    cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T, device=device)
    rend = renderer(point_cloud, cameras=cameras)
    rend = rend.cpu().numpy()[0, ..., :3]  # (B, H, W, 4) -> (H, W, 3)
    return rend


def render_sphere(image_size=256, num_samples=200, device=None):
    """
    Renders a sphere using parametric sampling. Samples num_samples ** 2 points.
    """

    if device is None:
        device = get_device()

    phi = torch.linspace(0, 2 * np.pi, num_samples)
    theta = torch.linspace(0, np.pi, num_samples)
    # Densely sample phi and theta on a grid
    Phi, Theta = torch.meshgrid(phi, theta)

    x = torch.sin(Theta) * torch.cos(Phi)
    y = torch.cos(Theta)
    z = torch.sin(Theta) * torch.sin(Phi)

    points = torch.stack((x.flatten(), y.flatten(), z.flatten()), dim=1)
    color = (points - points.min()) / (points.max() - points.min())

    sphere_point_cloud = pytorch3d.structures.Pointclouds(
        points=[points], features=[color],
    ).to(device)

    cameras = pytorch3d.renderer.FoVPerspectiveCameras(T=[[0, 0, 3]], device=device)
    renderer = get_points_renderer(image_size=image_size, device=device)
    rend = renderer(sphere_point_cloud, cameras=cameras)
    return rend[0, ..., :3].cpu().numpy()


def render_sphere_mesh(image_size=256, voxel_size=64, device=None):
    if device is None:
        device = get_device()
    min_value = -1.1
    max_value = 1.1
    X, Y, Z = torch.meshgrid([torch.linspace(min_value, max_value, voxel_size)] * 3)
    voxels = X ** 2 + Y ** 2 + Z ** 2 - 1
    vertices, faces = mcubes.marching_cubes(mcubes.smooth(voxels), isovalue=0)
    vertices = torch.tensor(vertices).float()
    faces = torch.tensor(faces.astype(int))
    # Vertex coordinates are indexed by array position, so we need to
    # renormalize the coordinate system.
    vertices = (vertices / voxel_size) * (max_value - min_value) + min_value
    textures = (vertices - vertices.min()) / (vertices.max() - vertices.min())
    textures = pytorch3d.renderer.TexturesVertex(vertices.unsqueeze(0))

    mesh = pytorch3d.structures.Meshes([vertices], [faces], textures=textures).to(
        device
    )
    lights = pytorch3d.renderer.PointLights(location=[[0, 0.0, -4.0]], device=device,)
    renderer = get_mesh_renderer(image_size=image_size, device=device)
    R, T = pytorch3d.renderer.look_at_view_transform(dist=3, elev=0, azim=180)
    cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T, device=device)
    rend = renderer(mesh, cameras=cameras, lights=lights)
    return rend[0, ..., :3].detach().cpu().numpy().clip(0, 1)


def render_plant_point_clouds(
    image_size=256,
    background_color=(1, 1, 1),
    num_frames=36,
    dist=6,
    output_path="images/plant_point_clouds.gif",
    device=None,
):
    """
    5.1 Renders the 2 plant RGB-D point clouds and their union, side by side,
    as a 360-degree turntable gif.
    """
    if device is None:
        device = get_device()
    data = load_rgbd_data()

    points1, rgba1 = unproject_depth_image(
        torch.tensor(data["rgb1"]),
        torch.tensor(data["mask1"]),
        torch.tensor(data["depth1"]),
        data["cameras1"],
    )
    points2, rgba2 = unproject_depth_image(
        torch.tensor(data["rgb2"]),
        torch.tensor(data["mask2"]),
        torch.tensor(data["depth2"]),
        data["cameras2"],
    )

    points_union = torch.cat([points1, points2], dim=0)
    rgba_union = torch.cat([rgba1, rgba2], dim=0)

    clouds = {
        "plant1": (points1, rgba1),
        "plant2": (points2, rgba2),
        "plant_union": (points_union, rgba_union),
    }
    point_clouds = {
        name: pytorch3d.structures.Pointclouds(
            points=points.unsqueeze(0), features=rgba[..., :3].unsqueeze(0)
        ).to(device)
        for name, (points, rgba) in clouds.items()
    }

    renderer = get_points_renderer(
        image_size=image_size, background_color=background_color, device=device
    )

    azimuths = torch.linspace(0, 360, num_frames + 1)[:-1]
    frames = []
    for azim in azimuths:
        # This dataset's world coordinates treat -Y as "up" (opposite of
        # PyTorch3D's default +Y), so the up vector needs to be flipped to
        # render the plant right-side-up instead of upside-down.
        R, T = pytorch3d.renderer.look_at_view_transform(
            dist=dist, elev=0, azim=azim.item(), up=((0, -1, 0),)
        )
        cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T, device=device)
        panels = []
        for name in ["plant1", "plant2", "plant_union"]:
            rend = renderer(point_clouds[name], cameras=cameras)
            panels.append(rend[0, ..., :3].cpu().numpy().clip(0, 1))
        frame = (np.concatenate(panels, axis=1) * 255).astype(np.uint8)
        frames.append(frame)

    imageio.mimsave(output_path, frames, duration=1000 // 15, loop=0)
    return output_path


def render_torus_point_cloud(
    image_size=256,
    num_samples=200,
    R=2.0,
    r=0.7,
    num_frames=36,
    dist=7,
    elev=30,
    output_path="images/torus.gif",
    device=None,
):
    """
    5.2 Renders a torus point cloud, sampled from its parametric equation, as
    a 360-degree turntable gif (elev != 0 so the hole stays visible).
    """
    if device is None:
        device = get_device()

    phi = torch.linspace(0, 2 * np.pi, num_samples)
    theta = torch.linspace(0, 2 * np.pi, num_samples)
    Phi, Theta = torch.meshgrid(phi, theta, indexing="ij")

    x = (R + r * torch.cos(Theta)) * torch.cos(Phi)
    y = (R + r * torch.cos(Theta)) * torch.sin(Phi)
    z = r * torch.sin(Theta)

    points = torch.stack((x.flatten(), y.flatten(), z.flatten()), dim=1)
    # Color by tube angle (theta) so the tube's cross-section reads clearly.
    color = torch.stack(
        [
            0.5 + 0.5 * torch.cos(Theta.flatten()),
            0.5 + 0.5 * torch.sin(Theta.flatten()),
            0.5 + 0.5 * torch.cos(Phi.flatten()),
        ],
        dim=1,
    )

    torus_point_cloud = pytorch3d.structures.Pointclouds(
        points=[points], features=[color]
    ).to(device)

    renderer = get_points_renderer(image_size=image_size, device=device)
    render_turntable_gif(
        torus_point_cloud,
        renderer,
        output_path,
        num_frames=num_frames,
        dist=dist,
        elev=elev,
        device=device,
    )
    return output_path


def render_spring_point_cloud(
    image_size=256,
    num_samples=100,
    R=1.5,
    r=0.4,
    turns=4,
    pitch=0.6,
    num_frames=36,
    dist=14,
    elev=20,
    output_path="images/spring.gif",
    device=None,
):
    """
    
    """
    if device is None:
        device = get_device()

    phi = torch.linspace(0, 2 * np.pi * turns, num_samples * turns)
    theta = torch.linspace(0, 2 * np.pi, num_samples)
    Phi, Theta = torch.meshgrid(phi, theta, indexing="ij")

    x = (R + r * torch.cos(Theta)) * torch.cos(Phi)
    y = (R + r * torch.cos(Theta)) * torch.sin(Phi)
    z = r * torch.sin(Theta) + pitch * Phi

    points = torch.stack((x.flatten(), y.flatten(), z.flatten()), dim=1)
    z_flat = points[:, 2]
    alpha = (z_flat - z_flat.min()) / (z_flat.max() - z_flat.min())
    color = torch.stack(
        [alpha, 1 - alpha, 0.5 + 0.5 * torch.cos(Theta.flatten())], dim=1
    )

    spring_point_cloud = pytorch3d.structures.Pointclouds(
        points=[points], features=[color]
    ).to(device)

    renderer = get_points_renderer(image_size=image_size, device=device)
    render_turntable_gif(
        spring_point_cloud,
        renderer,
        output_path,
        num_frames=num_frames,
        dist=dist,
        elev=elev,
        device=device,
    )
    return output_path


def render_torus_mesh(
    image_size=256,
    voxel_size=64,
    R=2.0,
    r=0.7,
    num_frames=36,
    dist=7,
    elev=30,
    output_path="images/torus_mesh.gif",
    device=None,
):
    """
    5.3 Renders a torus mesh, extracted via marching cubes from its implicit
    function F(x,y,z) = (sqrt(x^2+y^2) - R)^2 + z^2 - r^2, as a 360-degree
    turntable gif.
    """
    if device is None:
        device = get_device()
    min_value, max_value = -(R + r + 0.5), (R + r + 0.5)
    X, Y, Z = torch.meshgrid(
        [torch.linspace(min_value, max_value, voxel_size)] * 3, indexing="ij"
    )
    voxels = (torch.sqrt(X ** 2 + Y ** 2) - R) ** 2 + Z ** 2 - r ** 2
    vertices, faces = mcubes.marching_cubes(mcubes.smooth(voxels), isovalue=0)
    vertices = torch.tensor(vertices).float()
    faces = torch.tensor(faces.astype(int))
    vertices = (vertices / voxel_size) * (max_value - min_value) + min_value
    textures = (vertices - vertices.min()) / (vertices.max() - vertices.min())
    textures = pytorch3d.renderer.TexturesVertex(textures.unsqueeze(0))

    mesh = pytorch3d.structures.Meshes([vertices], [faces], textures=textures).to(device)
    lights = pytorch3d.renderer.PointLights(location=[[0, 0, -4.0]], device=device)
    renderer = get_mesh_renderer(image_size=image_size, device=device)
    render_turntable_gif(
        mesh, renderer, output_path, num_frames=num_frames, dist=dist, elev=elev,
        device=device, lights=lights,
    )
    return output_path


def render_metaballs_mesh(
    image_size=256,
    voxel_size=64,
    num_frames=36,
    dist=6,
    elev=20,
    output_path="images/metaballs.gif",
    device=None,
):
    """
    5.3  Renders a "metaballs" mesh: the smoothly-blended union
    of 3 implicit spheres, extracted via marching cubes, as a 360-degree
    turntable gif.
    """
    if device is None:
        device = get_device()
    centers = torch.tensor([[-0.8, 0.0, 0.0], [0.8, 0.0, 0.0], [0.0, 1.1, 0.3]])
    radii = torch.tensor([0.9, 0.9, 0.7])

    min_value, max_value = -2.2, 2.2
    X, Y, Z = torch.meshgrid(
        [torch.linspace(min_value, max_value, voxel_size)] * 3, indexing="ij"
    )
    field = torch.zeros_like(X)
    for center, radius in zip(centers, radii):
        dist_sq = (X - center[0]) ** 2 + (Y - center[1]) ** 2 + (Z - center[2]) ** 2
        field = field + (radius ** 2) / (dist_sq + 1e-6)
    voxels = 1.0 - field  # zero level-set is where the blobs blend together

    vertices, faces = mcubes.marching_cubes(mcubes.smooth(voxels), isovalue=0)
    vertices = torch.tensor(vertices).float()
    faces = torch.tensor(faces.astype(int))
    vertices = (vertices / voxel_size) * (max_value - min_value) + min_value
    textures = (vertices - vertices.min()) / (vertices.max() - vertices.min())
    textures = pytorch3d.renderer.TexturesVertex(textures.unsqueeze(0))

    mesh = pytorch3d.structures.Meshes([vertices], [faces], textures=textures).to(device)
    lights = pytorch3d.renderer.PointLights(location=[[0, 0, -4.0]], device=device)
    renderer = get_mesh_renderer(image_size=image_size, device=device)
    render_turntable_gif(
        mesh, renderer, output_path, num_frames=num_frames, dist=dist, elev=elev,
        device=device, lights=lights,
    )
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--render",
        type=str,
        default="point_cloud",
        choices=[
            "point_cloud",
            "parametric",
            "implicit",
            "plant_point_clouds",
            "torus_point_cloud",
            "spring_point_cloud",
            "torus_mesh",
            "metaballs_mesh",
        ],
    )
    parser.add_argument("--output_path", type=str, default="images/bridge.jpg")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--num_samples", type=int, default=100)
    args = parser.parse_args()
    if args.render == "point_cloud":
        image = render_bridge(image_size=args.image_size)
        plt.imsave(args.output_path, image)
    elif args.render == "parametric":
        image = render_sphere(image_size=args.image_size, num_samples=args.num_samples)
        plt.imsave(args.output_path, image)
    elif args.render == "implicit":
        image = render_sphere_mesh(image_size=args.image_size)
        plt.imsave(args.output_path, image)
    elif args.render == "plant_point_clouds":
        render_plant_point_clouds(
            image_size=args.image_size,
            output_path="images/plant_point_clouds.gif"
            if args.output_path == "images/bridge.jpg"
            else args.output_path,
        )
    elif args.render == "torus_point_cloud":
        render_torus_point_cloud(
            image_size=args.image_size,
            num_samples=args.num_samples,
            output_path="images/torus.gif"
            if args.output_path == "images/bridge.jpg"
            else args.output_path,
        )
    elif args.render == "spring_point_cloud":
        render_spring_point_cloud(
            image_size=args.image_size,
            output_path="images/spring.gif"
            if args.output_path == "images/bridge.jpg"
            else args.output_path,
        )
    elif args.render == "torus_mesh":
        render_torus_mesh(
            image_size=args.image_size,
            output_path="images/torus_mesh.gif"
            if args.output_path == "images/bridge.jpg"
            else args.output_path,
        )
    elif args.render == "metaballs_mesh":
        render_metaballs_mesh(
            image_size=args.image_size,
            output_path="images/metaballs.gif"
            if args.output_path == "images/bridge.jpg"
            else args.output_path,
        )
    else:
        raise Exception("Did not understand {}".format(args.render))

