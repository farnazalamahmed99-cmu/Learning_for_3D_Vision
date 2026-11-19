import os
import torch
import imageio
import argparse
import numpy as np

from PIL import Image
from tqdm import tqdm
from model import Scene, Gaussians
from torch.utils.data import DataLoader
from data_utils import visualize_renders
from data_utils_harder_scene import get_nerf_datasets, trivial_collate

from pytorch3d.renderer import PerspectiveCameras
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

def make_trainable(gaussians):

    ### YOUR CODE HERE ###
    # HINT: You can access and modify parameters from gaussians
    gaussians.means.requires_grad_(True)
    gaussians.pre_act_scales.requires_grad_(True)
    gaussians.colours.requires_grad_(True)
    gaussians.pre_act_opacities.requires_grad_(True)

    # Rotations only matter for anisotropic Gaussians
    if not gaussians.is_isotropic:
        gaussians.pre_act_quats.requires_grad_(True)

def setup_optimizer(gaussians, means_lr=0.0005):

    gaussians.check_if_trainable()

    ### YOUR CODE HERE ###
    # HINT: Modify the learning rates to reasonable values. We have intentionally
    # set very high learning rates for all parameters.
    # HINT: Consider reducing the learning rates for parameters that seem to vary too
    # fast with the default settings.
    # HINT: Consider setting different learning rates for different sets of parameters.
    parameters = [
        {'params': [gaussians.pre_act_opacities], 'lr': 0.05, "name": "opacities"},
        {'params': [gaussians.pre_act_scales], 'lr': 0.005, "name": "scales"},
        {'params': [gaussians.colours], 'lr': 0.02, "name": "colours"},
        {'params': [gaussians.means], 'lr': means_lr, "name": "means"},
    ]
    if not gaussians.is_isotropic:
        parameters.append({'params': [gaussians.pre_act_quats], 'lr': 0.001, "name": "quats"})
    optimizer = torch.optim.Adam(parameters, lr=0.0, eps=1e-15)

    return optimizer

def ssim_loss(pred, gt, window_size=11, sigma=1.5):
    """
    1 - SSIM between two (H, W, 3) images in [0, 1], using a Gaussian window
    (the same formulation as skimage / the 3D Gaussian Splatting paper).
    """
    x = pred.permute(2, 0, 1)[None]  # (1, 3, H, W)
    y = gt.permute(2, 0, 1)[None]

    coords = torch.arange(window_size, dtype=x.dtype, device=x.device) - window_size // 2
    g = torch.exp(-coords ** 2 / (2 * sigma ** 2))
    g = g / g.sum()
    window = (g[:, None] * g[None, :]).expand(3, 1, window_size, window_size).contiguous()
    blur = lambda t: torch.nn.functional.conv2d(t, window, padding=window_size // 2, groups=3)

    mu_x, mu_y = blur(x), blur(y)
    var_x = blur(x * x) - mu_x ** 2
    var_y = blur(y * y) - mu_y ** 2
    cov_xy = blur(x * y) - mu_x * mu_y

    C1, C2 = 0.01 ** 2, 0.03 ** 2
    ssim = ((2 * mu_x * mu_y + C1) * (2 * cov_xy + C2)) / ((mu_x ** 2 + mu_y ** 2 + C1) * (var_x + var_y + C2))
    return 1.0 - ssim.mean()

def ndc_to_screen_camera(camera, img_size = (128, 128)):

    min_size = min(img_size[0], img_size[1])

    screen_focal = camera.focal_length * min_size / 2.0
    screen_principal = torch.tensor([[img_size[0]/2, img_size[1]/2]]).to(torch.float32)

    return PerspectiveCameras(
        R=camera.R, T=camera.T, in_ndc=False,
        focal_length=screen_focal, principal_point=screen_principal,
        image_size=(img_size,),
    )

def run_training(args):

    if not os.path.exists(args.out_path):
        os.makedirs(args.out_path, exist_ok=True)

    train_dataset, val_dataset, _ = get_nerf_datasets(
        dataset_name="materials", data_root=args.data_path,
        image_size=[128, 128],
    )

    train_loader = DataLoader(
        train_dataset, batch_size=1, shuffle=True, num_workers=0,
        drop_last=True, collate_fn=trivial_collate
    )
    val_loader = DataLoader(
        val_dataset, batch_size=1, shuffle=False, num_workers=0,
        drop_last=True, collate_fn=trivial_collate
    )
    train_itr = iter(train_loader)

    # Preparing some code for visualization
    viz_gif_path_1 = os.path.join(args.out_path, "part_7_2_training_progress.gif")
    viz_gif_path_2 = os.path.join(args.out_path, "part_7_2_training_final_renders.gif")
    viz_idxs = np.linspace(0, len(train_dataset)-1, 5).astype(np.int32)[:4]

    gt_viz_imgs = [(train_dataset[i]["image"]*255.0).numpy().astype(np.uint8) for i in viz_idxs]
    gt_viz_imgs = [np.array(Image.fromarray(x).resize((256, 256))) for x in gt_viz_imgs]
    gt_viz_img = np.concatenate(gt_viz_imgs, axis=1)

    viz_cameras = [ndc_to_screen_camera(train_dataset[i]["camera"]).cuda() for i in viz_idxs]

    # Init gaussians and scene
    gaussians = Gaussians(
        num_points=10000, init_type="random",
        device=args.device, isotropic=not args.anisotropic
    )

    # The default random init places all means in a ball of std 0.2, much smaller
    # than the scene; optionally rescale it so Gaussians start closer to the objects
    if args.init_std != 0.2:
        gaussians.means = gaussians.means * (args.init_std / 0.2)

    scene = Scene(gaussians)

    # Making gaussians trainable and setting up optimizer
    make_trainable(gaussians)
    optimizer = setup_optimizer(gaussians, means_lr=args.means_lr)

    # Optional exponential decay of the means learning rate (to 1% of its initial
    # value by the end of training); all other learning rates stay constant
    lr_lambdas = [
        (lambda itr: 0.01 ** (itr / args.num_itrs)) if (group["name"] == "means" and args.lr_decay) else (lambda itr: 1.0)
        for group in optimizer.param_groups
    ]
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambdas)

    # Training loop
    viz_frames = []
    for itr in range(args.num_itrs):

        # Fetching data
        try:
            data = next(train_itr)
        except StopIteration:
            train_itr = iter(train_loader)
            data = next(train_itr)

        gt_img = data[0]["image"].cuda()
        camera = ndc_to_screen_camera(data[0]["camera"]).cuda()

        # Rendering scene using gaussian splatting
        ### YOUR CODE HERE ###
        # HINT: Can any function from the Scene class help?
        # HINT: Set bg_colour to (0.0, 0.0, 0.0)
        # HINT: Set img_size to (128, 128)
        # HINT: Get per_splat from args.gaussians_per_splat
        # HINT: camera is available above
        pred_img, _, _ = scene.render(
            camera, per_splat=args.gaussians_per_splat,
            img_size=(128, 128), bg_colour=(0.0, 0.0, 0.0),
        )

        # Compute loss
        ### YOUR CODE HERE ###
        loss = torch.abs(pred_img - gt_img).mean()
        if args.ssim_weight > 0:
            loss = (1.0 - args.ssim_weight) * loss + args.ssim_weight * ssim_loss(pred_img, gt_img)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        lr_scheduler.step()

        print(f"[*] Itr: {itr:07d} | Loss: {loss:0.3f}")

        if itr % args.viz_freq == 0:
            viz_frame = visualize_renders(
                scene, gt_viz_img,
                viz_cameras, (128, 128)
            )
            viz_frames.append(viz_frame)

    print("[*] Training Completed.")

    # Saving training progess GIF
    imageio.mimwrite(viz_gif_path_1, viz_frames, loop=0, duration=(1/10.0)*1000)

    # Creating renderings of the training views after training is completed.
    frames = []
    viz_loader = DataLoader(
        train_dataset, batch_size=1, shuffle=False, num_workers=0,
        drop_last=True, collate_fn=trivial_collate
    )
    for viz_data in tqdm(viz_loader, desc="Creating Visualization"):
        gt_img = viz_data[0]["image"].cuda()
        camera = ndc_to_screen_camera(viz_data[0]["camera"]).cuda()

        with torch.no_grad():

            # Rendering scene using gaussian splatting
            ### YOUR CODE HERE ###
            # HINT: Can any function from the Scene class help?
            # HINT: Set bg_colour to (0.0, 0.0, 0.0)
            # HINT: Set img_size to (128, 128)
            # HINT: Get per_splat from args.gaussians_per_splat
            # HINT: camera is available above
            pred_img, _, _ = scene.render(
                camera, per_splat=args.gaussians_per_splat,
                img_size=(128, 128), bg_colour=(0.0, 0.0, 0.0),
            )

        pred_npy = pred_img.detach().cpu().numpy()
        pred_npy = (np.clip(pred_npy, 0.0, 1.0) * 255.0).astype(np.uint8)
        frames.append(pred_npy)

    # Saving renderings
    imageio.mimwrite(viz_gif_path_2, frames, loop=0, duration=(1/10.0)*1000)

    # Running evaluation using the test dataset
    psnr_vals, ssim_vals = [], []
    for val_data in tqdm(val_loader, desc="Running Evaluation"):

        gt_img = val_data[0]["image"].cuda()
        camera = ndc_to_screen_camera(val_data[0]["camera"]).cuda()

        with torch.no_grad():

            # Rendering scene using gaussian splatting
            # Rendering scene using gaussian splatting
            ### YOUR CODE HERE ###
            # HINT: Can any function from the Scene class help?
            # HINT: Set bg_colour to (0.0, 0.0, 0.0)
            # HINT: Set img_size to (128, 128)
            # HINT: Get per_splat from args.gaussians_per_splat
            # HINT: camera is available above
            pred_img, _, _ = scene.render(
                camera, per_splat=args.gaussians_per_splat,
                img_size=(128, 128), bg_colour=(0.0, 0.0, 0.0),
            )

            gt_npy = gt_img.detach().cpu().numpy()
            pred_npy = pred_img.detach().cpu().numpy()
            psnr = peak_signal_noise_ratio(gt_npy, pred_npy)
            ssim = structural_similarity(gt_npy, pred_npy, channel_axis=-1, data_range=1.0)

            psnr_vals.append(psnr)
            ssim_vals.append(ssim)

    mean_psnr = np.mean(psnr_vals)
    mean_ssim = np.mean(ssim_vals)
    print(f"[*] Evaluation --- Mean PSNR: {mean_psnr:.3f}")
    print(f"[*] Evaluation --- Mean SSIM: {mean_ssim:.3f}")

def get_args():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out_path", default="./output", type=str,
        help="Path to the directory where output should be saved to."
    )
    parser.add_argument(
        "--data_path", default="./data/materials", type=str,
        help="Path to the dataset."
    )
    parser.add_argument(
        "--gaussians_per_splat", default=-1, type=int,
        help=(
            "Number of gaussians to splat in one function call. If set to -1, "
            "then all gaussians in the scene are splat in a single function call. "
            "If set to any other positive interger, then it determines the number of "
            "gaussians to splat per function call (the last function call might splat "
            "lesser number of gaussians). In general, the algorithm can run faster "
            "if more gaussians are splat per function call, but at the cost of higher GPU "
            "memory consumption."
        )
    )
    parser.add_argument(
        "--num_itrs", default=1000, type=int,
        help="Number of iterations to train the model."
    )
    parser.add_argument(
        "--viz_freq", default=20, type=int,
        help="Frequency with which visualization should be performed."
    )
    # Improvements over the Q6 baseline (the defaults reproduce the baseline)
    parser.add_argument("--anisotropic", action="store_true", help="Train anisotropic Gaussians (scales + rotations).")
    parser.add_argument("--init_std", default=0.2, type=float, help="Std of the random initial means.")
    parser.add_argument("--means_lr", default=0.0005, type=float, help="Initial learning rate of the means.")
    parser.add_argument("--lr_decay", action="store_true", help="Exponentially decay the means learning rate to 1%%.")
    parser.add_argument("--ssim_weight", default=0.0, type=float, help="Weight of the (1 - SSIM) loss term.")
    parser.add_argument("--device", default="cuda", type=str, choices=["cuda", "cpu"])
    args = parser.parse_args()
    return args

if __name__ == "__main__":

    args = get_args()
    run_training(args)
