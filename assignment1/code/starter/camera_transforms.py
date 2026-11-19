"""
Usage:
    python -m starter.camera_transforms --image_size 512
"""
import argparse

import matplotlib.pyplot as plt
import pytorch3d
import torch

from starter.utils import get_device, get_mesh_renderer, load_cow_mesh


def render_textured_cow(
    cow_path="data/cow.obj",
    image_size=256,
    color=[0.7, 0.7, 1],
    R_relative=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    T_relative=[0, 0, 0],
    device=None,
):
    if device is None:
        device = get_device()
    vertices, faces = load_cow_mesh(cow_path)
    textures = torch.ones_like(vertices) * torch.tensor(color)
    meshes = pytorch3d.structures.Meshes(
        verts=vertices.unsqueeze(0),
        faces=faces.unsqueeze(0),
        textures=pytorch3d.renderer.TexturesVertex(textures.unsqueeze(0)),
    ).to(device)
    R_relative = torch.tensor(R_relative).float()
    T_relative = torch.tensor(T_relative).float()
    R = R_relative @ torch.tensor([[1.0, 0, 0], [0, 1, 0], [0, 0, 1]])
    T = R_relative @ torch.tensor([0.0, 0, 3]) + T_relative
    renderer = get_mesh_renderer(image_size=256)
    cameras = pytorch3d.renderer.FoVPerspectiveCameras(
        R=R.unsqueeze(0), T=T.unsqueeze(0), device=device,
    )
    lights = pytorch3d.renderer.PointLights(location=[[0, 0.0, -3.0]], device=device,)
    rend = renderer(meshes, cameras=cameras, lights=lights)
    return rend[0, ..., :3].cpu().numpy()

TRANSFORMS = {
    # Roll the camera -90 degrees about its own viewing (Z) axis: the camera
    # stays in place (T unchanged) but the image rotates in-plane, so the cow
    # appears to fall onto its side.
    "transform1": (
        [[0, 1, 0], [-1, 0, 0], [0, 0, 1]],  # R_relative
        [0, 0, 0],  # T_relative
    ),
    # Orbit the camera 90 degrees about the world Y (up) axis so it ends up
    # at world position (-3, 0, 0) looking back at the origin (same
    # distance), to view the cow from the side.
    "transform2": (
        [[0, 0, 1], [0, 1, 0], [-1, 0, 0]],
        [-3, 0, 3],
    ),
    # Move the camera straight back (increase distance along its viewing
    # axis, 3 -> 5) to "zoom out": no rotation, just a +z translation.
    "transform3": (
        [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        [0, 0, 2],
    ),
    # Slide the camera sideways (translate along world X): no rotation, just
    # a small x translation.
    "transform4": (
        [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        [0.5, 0, 0],
    ),
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cow_path", type=str, default="data/cow.obj")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--output_path", type=str, default="images/textured_cow.jpg")
    parser.add_argument(
        "--transform",
        type=str,
        default=None,
        choices=list(TRANSFORMS.keys()),
        help="If set, render just this one named transform instead of the default view.",
    )
    args = parser.parse_args()

    if args.transform is not None:
        R_relative, T_relative = TRANSFORMS[args.transform]
        image = render_textured_cow(
            cow_path=args.cow_path,
            image_size=args.image_size,
            R_relative=R_relative,
            T_relative=T_relative,
        )
        plt.imsave(args.output_path, image)
    else:
        
        for name, (R_relative, T_relative) in TRANSFORMS.items():
            image = render_textured_cow(
                cow_path=args.cow_path,
                image_size=args.image_size,
                R_relative=R_relative,
                T_relative=T_relative,
            )
            plt.imsave(f"images/{name}.jpg", image)
