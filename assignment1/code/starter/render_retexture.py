"""
3. Re-textures the cow mesh with a color gradient from the front (min z) to
the back (max z), and renders a 360-degree turntable gif.

Usage:
    python -m starter.render_retexture --output_path images/cow_retextured.gif
"""
import argparse

import imageio
import numpy as np
import pytorch3d
import torch

from starter.utils import get_device, get_mesh_renderer, load_cow_mesh


def get_retextured_cow_mesh(
    cow_path="data/cow.obj", color1=[0, 0, 1], color2=[1, 0, 0], device=None
):
    if device is None:
        device = get_device()

    vertices, faces = load_cow_mesh(cow_path)
    color1 = torch.tensor(color1, dtype=torch.float32)
    color2 = torch.tensor(color2, dtype=torch.float32)

    
    z=vertices[:, 2]
    z_min=z.min()
    z_max=z.max()
    alpha=((z-z_min)/(z_max-z_min)).unsqueeze(1)
    color=alpha*color2+(1-alpha)*color1
    textures = color

    mesh = pytorch3d.structures.Meshes(
        verts=vertices.unsqueeze(0),
        faces=faces.unsqueeze(0),
        textures=pytorch3d.renderer.TexturesVertex(textures.unsqueeze(0)),
    )
    return mesh.to(device)


def render_retextured_cow_360(
    cow_path="data/cow.obj",
    color1=[0, 0, 1],
    color2=[1, 0, 0],
    image_size=256,
    num_frames=36,
    dist=3,
    elev=0,
    fps=15,
    output_path="images/cow_retextured.gif",
    device=None,
):
    if device is None:
        device = get_device()

    mesh = get_retextured_cow_mesh(cow_path, color1, color2, device)

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cow_path", type=str, default="data/cow.obj")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--num_frames", type=int, default=36)
    parser.add_argument("--output_path", type=str, default="images/cow_retextured.gif")
    args = parser.parse_args()
    render_retextured_cow_360(
        cow_path=args.cow_path,
        image_size=args.image_size,
        num_frames=args.num_frames,
        output_path=args.output_path,
    )
