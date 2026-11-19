import imageio
import numpy as np
import pytorch3d
import torch
from pytorch3d.renderer import (
    AlphaCompositor,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    PointsRasterizationSettings,
    PointsRenderer,
    PointsRasterizer,
    HardPhongShader,
)
from pytorch3d.io import load_obj
import mcubes
import pytorch3d.renderer
import pytorch3d.structures

from utils_vox import Mem2Ref


def get_device():
    """
    Checks if GPU is available and returns device accordingly.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    return device


def get_points_renderer(
    image_size=512, device=None, radius=0.01, background_color=(1, 1, 1)
):
    """
    Returns a Pytorch3D renderer for point clouds.

    Args:
        image_size (int): The rendered image size.
        device (torch.device): The torch device to use (CPU or GPU). If not specified,
            will automatically use GPU if available, otherwise CPU.
        radius (float): The radius of the rendered point in NDC.
        background_color (tuple): The background color of the rendered image.
    
    Returns:
        PointsRenderer.
    """
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")
    raster_settings = PointsRasterizationSettings(image_size=image_size, radius=radius,)
    renderer = PointsRenderer(
        rasterizer=PointsRasterizer(raster_settings=raster_settings),
        compositor=AlphaCompositor(background_color=background_color),
    )
    return renderer


def get_mesh_renderer(image_size=512, lights=None, device=None):
    """
    Returns a Pytorch3D Mesh Renderer.

    Args:
        image_size (int): The rendered image size.
        lights: A default Pytorch3D lights object.
        device (torch.device): The torch device to use (CPU or GPU). If not specified,
            will automatically use GPU if available, otherwise CPU.
    """
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")
    raster_settings = RasterizationSettings(
        image_size=image_size, blur_radius=0.0, faces_per_pixel=1,
    )
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(raster_settings=raster_settings),
        shader=HardPhongShader(device=device, lights=lights),
    )
    return renderer


def unproject_depth_image(image, mask, depth, camera):
    """
    Unprojects a depth image into a 3D point cloud.

    Args:
        image (torch.Tensor): A square image to unproject (S, S, 3).
        mask (torch.Tensor): A binary mask for the image (S, S).
        depth (torch.Tensor): The depth map of the image (S, S).
        camera: The Pytorch3D camera to render the image.
    
    Returns:
        points (torch.Tensor): The 3D points of the unprojected image (N, 3).
        rgba (torch.Tensor): The rgba color values corresponding to the unprojected
            points (N, 4).
    """
    device = camera.device
    assert image.shape[0] == image.shape[1], "Image must be square."
    image_shape = image.shape[0]
    ndc_pixel_coordinates = torch.linspace(1, -1, image_shape)
    Y, X = torch.meshgrid(ndc_pixel_coordinates, ndc_pixel_coordinates)
    xy_depth = torch.dstack([X, Y, depth])
    points = camera.unproject_points(
        xy_depth.to(device), in_ndc=False, from_ndc=False, world_coordinates=True,
    )
    points = points[mask > 0.5]
    rgb = image[mask > 0.5]
    rgb = rgb.to(device)

    # For some reason, the Pytorch3D compositor does not apply a background color
    # unless the pointcloud is RGBA.
    alpha = torch.ones_like(rgb)[..., :1]
    rgb = torch.cat([rgb, alpha], dim=1)

    return points, rgb


def render_turntable_gif(
    renderable,
    renderer,
    output_path,
    num_frames=36,
    dist=3,
    elev=0,
    fps=15,
    device=None,
    lights=None,
    loop=0,
):
    """
    Renders a 360-degree turntable gif of a mesh or point cloud.

    Args:
        renderable: A Pytorch3D Meshes or Pointclouds object (already on `device`).
        renderer: A Pytorch3D renderer (from get_mesh_renderer / get_points_renderer)
            compatible with `renderable`.
        output_path (str): Where to save the resulting gif.
        num_frames (int): Number of frames in the turntable.
        dist (float): Camera distance from the origin.
        elev (float): Camera elevation in degrees.
        fps (int): Frames per second of the output gif.
        device: The torch device to render on.
        lights: Optional Pytorch3D lights object (only used for mesh rendering).
        loop (int): Number of times the gif should loop (0 = loop forever).

    Returns:
        images (List[np.ndarray]): The list of rendered frames (H, W, 3) uint8.
    """
    if device is None:
        device = get_device()

    azimuths = torch.linspace(0, 360, num_frames + 1)[:-1]
    images = []
    for azim in azimuths:
        R, T = pytorch3d.renderer.look_at_view_transform(
            dist=dist, elev=elev, azim=azim.item()
        )
        cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T, device=device)
        if lights is not None:
            rend = renderer(renderable, cameras=cameras, lights=lights)
        else:
            rend = renderer(renderable, cameras=cameras)
        image = rend[0, ..., :3].detach().cpu().numpy().clip(0, 1)
        images.append((image * 255).astype(np.uint8))

    duration = 1000 // fps
    imageio.mimsave(output_path, images, duration=duration, loop=loop)
    return images


def save_side_by_side_gif(frame_lists, output_path, fps=15, loop=0):
    """
    Places several turntable gifs next to each other, frame by frame.

    Args:
        frame_lists: A list of frame lists, as returned by render_turntable_gif.
            All must have the same number of frames and image size.
        output_path (str): Where to save the combined gif.
        fps (int): Frames per second of the output gif.
        loop (int): Number of times the gif should loop (0 = loop forever).
    """
    images = [np.concatenate(frames, axis=1) for frames in zip(*frame_lists)]
    duration = 1000 // fps
    imageio.mimsave(output_path, images, duration=duration, loop=loop)
    return images


def load_cow_mesh(path="data/cow_mesh.obj"):
    """
    Loads vertices and faces from an obj file.

    Returns:
        vertices (torch.Tensor): The vertices of the mesh (N_v, 3).
        faces (torch.Tensor): The faces of the mesh (N_f, 3).
    """
    vertices, faces, _ = load_obj(path)
    faces = faces.verts_idx
    return vertices, faces


def voxels_to_mesh(voxels, device=None, isovalue=0.5, color=[0.7, 0.7, 1]):
    """
    Converts an occupancy voxel grid into a mesh via marching cubes.

    Args:
        voxels (torch.Tensor): Occupancy probabilities (Z, Y, X), as produced by
            utils_vox.voxelize_xyz. Values in [0, 1].
        device: The torch device to put the mesh on.
        isovalue (float): Occupancy level where the surface is placed.
        color (list): RGB color of the mesh.

    Returns:
        A Pytorch3D Meshes object in the same world coordinates as the R2N2 meshes.
    """
    if device is None:
        device = get_device()
    D = voxels.shape[-1]
    # Pad with empty cells so shapes touching the grid border still get a closed surface.
    grid = np.pad(voxels.detach().cpu().numpy(), 1)
    vertices, faces = mcubes.marching_cubes(grid, isovalue)
    vertices = torch.tensor(vertices).float() - 1  # undo the padding offset
    # Grid is indexed [Z, Y, X]; reorder to (x, y, z) and map cell indices to world coords.
    vertices = vertices[:, [2, 1, 0]]
    vertices = Mem2Ref(vertices.unsqueeze(0), D, D, D)[0]
    faces = torch.tensor(faces.astype(np.int64))
    textures = torch.ones_like(vertices).unsqueeze(0) * torch.tensor(color)
    mesh = pytorch3d.structures.Meshes(
        verts=[vertices],
        faces=[faces],
        textures=pytorch3d.renderer.TexturesVertex(textures),
    )
    return mesh.to(device)
