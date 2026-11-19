import argparse
import os
import time

import losses
from pytorch3d.utils import ico_sphere
from r2n2_custom import R2N2
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.structures import Meshes, Pointclouds
import dataset_location
import torch
import pytorch3d.renderer
from render_utils import get_mesh_renderer, get_points_renderer, render_turntable_gif, save_side_by_side_gif, voxels_to_mesh






def get_args_parser():
    parser = argparse.ArgumentParser('Model Fit', add_help=False)
    parser.add_argument('--lr', default=4e-4, type=float)
    parser.add_argument('--max_iter', default=100000, type=int)
    parser.add_argument('--type', default='vox', choices=['vox', 'point', 'mesh'], type=str)
    parser.add_argument('--n_points', default=5000, type=int)
    parser.add_argument('--w_chamfer', default=1.0, type=float)
    parser.add_argument('--w_smooth', default=0.1, type=float)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--output_dir', default='output', type=str)
    return parser

def fit_mesh(mesh_src, mesh_tgt, args):
    start_iter = 0
    start_time = time.time()

    deform_vertices_src = torch.zeros(mesh_src.verts_packed().shape, requires_grad=True, device='cuda')
    optimizer = torch.optim.Adam([deform_vertices_src], lr = args.lr)
    print("Starting training !")
    for step in range(start_iter, args.max_iter):
        iter_start_time = time.time()

        new_mesh_src = mesh_src.offset_verts(deform_vertices_src)

        sample_trg = sample_points_from_meshes(mesh_tgt, args.n_points)
        sample_src = sample_points_from_meshes(new_mesh_src, args.n_points)

        loss_reg = losses.chamfer_loss(sample_src, sample_trg)
        loss_smooth = losses.smoothness_loss(new_mesh_src)

        loss = args.w_chamfer * loss_reg + args.w_smooth * loss_smooth

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()        

        total_time = time.time() - start_time
        iter_time = time.time() - iter_start_time

        loss_vis = loss.cpu().item()

        print("[%4d/%4d]; ttime: %.0f (%.2f); loss: %.3f" % (step, args.max_iter, total_time,  iter_time, loss_vis))        
    
    mesh_src.offset_verts_(deform_vertices_src)

    print('Done!')


def fit_pointcloud(pointclouds_src, pointclouds_tgt, args):
    start_iter = 0
    start_time = time.time()    
    optimizer = torch.optim.Adam([pointclouds_src], lr = args.lr)
    for step in range(start_iter, args.max_iter):
        iter_start_time = time.time()

        loss = losses.chamfer_loss(pointclouds_src, pointclouds_tgt)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()        

        total_time = time.time() - start_time
        iter_time = time.time() - iter_start_time

        loss_vis = loss.cpu().item()

        print("[%4d/%4d]; ttime: %.0f (%.2f); loss: %.3f" % (step, args.max_iter, total_time,  iter_time, loss_vis))
    
    print('Done!')


def fit_voxel(voxels_src, voxels_tgt, args):
    start_iter = 0
    start_time = time.time()    
    optimizer = torch.optim.Adam([voxels_src], lr = args.lr)
    for step in range(start_iter, args.max_iter):
        iter_start_time = time.time()

        loss = losses.voxel_loss(voxels_src,voxels_tgt)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()        

        total_time = time.time() - start_time
        iter_time = time.time() - iter_start_time

        loss_vis = loss.cpu().item()

        print("[%4d/%4d]; ttime: %.0f (%.2f); loss: %.3f" % (step, args.max_iter, total_time,  iter_time, loss_vis))
    
    print('Done!')


def train_model(args):
    r2n2_dataset = R2N2("train", dataset_location.SHAPENET_PATH, dataset_location.R2N2_PATH, dataset_location.SPLITS_PATH, return_voxels=True)

    
    feed = r2n2_dataset[0]


    feed_cuda = {}
    for k in feed:
        if torch.is_tensor(feed[k]):
            feed_cuda[k] = feed[k].to(args.device).float()


    if args.type == "vox":
        # initialization
        voxels_src = torch.rand(feed_cuda['voxels'].shape,requires_grad=True, device=args.device)
        voxel_coords = feed_cuda['voxel_coords'].unsqueeze(0)
        voxels_tgt = feed_cuda['voxels']

        # fitting
        fit_voxel(voxels_src, voxels_tgt, args)

        # visualization: 360-degree gifs of the fitted and ground truth voxel grids
        os.makedirs(args.output_dir, exist_ok=True)
        renderer = get_mesh_renderer(image_size=256, device=args.device)
        lights = pytorch3d.renderer.PointLights(location=[[0, 0, -3]], device=args.device)
        mesh_src = voxels_to_mesh(torch.sigmoid(voxels_src)[0], device=args.device)
        mesh_tgt = voxels_to_mesh(voxels_tgt[0], device=args.device, color=[1, 0.7, 0.7])
        frames_src = render_turntable_gif(mesh_src, renderer, os.path.join(args.output_dir, 'q1_1_vox_src.gif'),
                                          dist=1.5, elev=20, lights=lights, device=args.device)
        frames_tgt = render_turntable_gif(mesh_tgt, renderer, os.path.join(args.output_dir, 'q1_1_vox_tgt.gif'),
                                          dist=1.5, elev=20, lights=lights, device=args.device)
        # fitted (left) next to ground truth (right)
        save_side_by_side_gif([frames_src, frames_tgt], os.path.join(args.output_dir, 'q1_1_vox.gif'))


    elif args.type == "point":
        # initialization
        pointclouds_src = torch.randn([1,args.n_points,3],requires_grad=True, device=args.device)
        mesh_tgt = Meshes(verts=[feed_cuda['verts']], faces=[feed_cuda['faces']])
        pointclouds_tgt = sample_points_from_meshes(mesh_tgt, args.n_points)

        # fitting
        fit_pointcloud(pointclouds_src, pointclouds_tgt, args)

        # visualization: 360-degree gifs of the fitted and ground truth point clouds
        os.makedirs(args.output_dir, exist_ok=True)
        renderer = get_points_renderer(image_size=256, radius=0.01, device=args.device)
        points_src = pointclouds_src.detach()
        # RGBA colors: the compositor only applies the white background to RGBA point clouds
        color_src = torch.tensor([0.4, 0.4, 1.0, 1.0], device=args.device).expand(1, args.n_points, 4)
        color_tgt = torch.tensor([1.0, 0.4, 0.4, 1.0], device=args.device).expand(1, args.n_points, 4)
        cloud_src = Pointclouds(points=points_src, features=color_src)
        cloud_tgt = Pointclouds(points=pointclouds_tgt, features=color_tgt)
        frames_src = render_turntable_gif(cloud_src, renderer, os.path.join(args.output_dir, 'q1_2_point_src.gif'),
                                          dist=1.5, elev=20, device=args.device)
        frames_tgt = render_turntable_gif(cloud_tgt, renderer, os.path.join(args.output_dir, 'q1_2_point_tgt.gif'),
                                          dist=1.5, elev=20, device=args.device)
        # fitted (left) next to ground truth (right)
        save_side_by_side_gif([frames_src, frames_tgt], os.path.join(args.output_dir, 'q1_2_point.gif'))
    
    elif args.type == "mesh":
        # initialization
        # try different ways of initializing the source mesh        
        mesh_src = ico_sphere(4, args.device)
        mesh_tgt = Meshes(verts=[feed_cuda['verts']], faces=[feed_cuda['faces']])

        # fitting
        fit_mesh(mesh_src, mesh_tgt, args)

        # visualization: 360-degree gifs of the fitted and ground truth meshes
        os.makedirs(args.output_dir, exist_ok=True)
        renderer = get_mesh_renderer(image_size=256, device=args.device)
        lights = pytorch3d.renderer.PointLights(location=[[0, 0, -3]], device=args.device)
        meshes_vis = []
        for mesh, color in ((mesh_src, [0.7, 0.7, 1]), (mesh_tgt, [1, 0.7, 0.7])):
            verts = mesh.verts_packed().detach()
            faces = mesh.faces_packed().long()
            textures = torch.ones_like(verts).unsqueeze(0) * torch.tensor(color, device=args.device)
            meshes_vis.append(Meshes(verts=[verts], faces=[faces],
                                     textures=pytorch3d.renderer.TexturesVertex(textures)))
        frames_src = render_turntable_gif(meshes_vis[0], renderer, os.path.join(args.output_dir, 'q1_3_mesh_src.gif'),
                                          dist=1.5, elev=20, lights=lights, device=args.device)
        frames_tgt = render_turntable_gif(meshes_vis[1], renderer, os.path.join(args.output_dir, 'q1_3_mesh_tgt.gif'),
                                          dist=1.5, elev=20, lights=lights, device=args.device)
        # fitted (left) next to ground truth (right)
        save_side_by_side_gif([frames_src, frames_tgt], os.path.join(args.output_dir, 'q1_3_mesh.gif'))


    
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Model Fit', parents=[get_args_parser()])
    args = parser.parse_args()
    train_model(args)
