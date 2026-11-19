from torchvision import models as torchvision_models
from torchvision import transforms
import time
import torch.nn as nn
import torch
from pytorch3d.utils import ico_sphere
import pytorch3d

class SingleViewto3D(nn.Module):
    def __init__(self, args):
        super(SingleViewto3D, self).__init__()
        self.device = args.device
        if not args.load_feat:
            vision_model = torchvision_models.__dict__[args.arch](pretrained=True)
            self.encoder = torch.nn.Sequential(*(list(vision_model.children())[:-1]))
            self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])


        # define decoder
        if args.type == "vox":
            # Input: b x 512
            # Output: b x 32 x 32 x 32
            self.fc=nn.Sequential(
                nn.Linear(512,128*4*4*4),
                nn.ReLU()
            )
        
            # TODO:
            self.decoder =   nn.Sequential(
                nn.ConvTranspose3d(128, 64, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm3d(64),
                nn.ReLU(),
                nn.ConvTranspose3d(64, 32, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm3d(32),
                nn.ReLU(),
                nn.ConvTranspose3d(32, 16, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm3d(16),
                nn.ReLU(),
                nn.ConvTranspose3d(16, 1, kernel_size=3, padding=1),
                
            )          
        elif args.type == "point":
            # Input: b x 512
            # Output: b x args.n_points x 3  
            self.n_point = args.n_points
            self.decoder=nn.Sequential(
                nn.Linear(512, 1024),
                nn.ReLU(),
                nn.Linear(1024,2048),
                nn.ReLU(),
                nn.Linear(2048, self.n_point * 3),
               
            )
            # TODO:
            # self.decoder =             
        elif args.type == "mesh":
            # Input: b x 512
            # Output: b x mesh_pred.verts_packed().shape[0] x 3  
            # try different mesh initializations
            mesh_pred = ico_sphere(4, self.device)
            self.mesh_pred = pytorch3d.structures.Meshes(mesh_pred.verts_list()*args.batch_size, mesh_pred.faces_list()*args.batch_size)
            self.n_vert = mesh_pred.verts_packed().shape[0]
            self.decoder=nn.Sequential(
                nn.Linear(512,1024),
                nn.ReLU(),
                nn.Linear(1024,2048),
                nn.ReLU(),
                nn.Linear(2048, self.n_vert * 3),
                
            )
            # TODO:
            # self.decoder =   
        elif args.type=="implicit":
            self.decoder = nn.Sequential(
                nn.Linear(512+3,256),
                nn.ReLU(), 
                nn.Linear(256,256),
                nn.ReLU(), 
                nn.Linear(256,256),
                nn.ReLU(), 
                nn.Linear(256,1),
            )
            # query points: a 32^3 grid in (-1, 1)^3, flattened in the same [Z, Y, X] order as the voxels
            lin=torch.linspace(-1,1,32)
            z, y, x = torch.meshgrid(lin, lin, lin, indexing="ij")
            self.register_buffer("grid", torch.stack([x, y, z], dim=-1).view(-1, 3))   # 32768 x 3
            self.n_train_queries = 4096
        elif args.type == "parametric":
            # Input: b x 512 image feature + one 2D sample (u, v) from the unit square
            # Output: the 3D surface point that (u, v) is mapped to
            self.n_point = args.n_points
            self.decoder = nn.Sequential(
                nn.Linear(512 + 2, 512),
                nn.ReLU(),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, 3),
                nn.Tanh(),
            )
                         

    def forward(self, images, args):
        results = dict()

        total_loss = 0.0
        start_time = time.time()

        B = images.shape[0]

        if not args.load_feat:
            images_normalize = self.normalize(images.permute(0,3,1,2))
            encoded_feat = self.encoder(images_normalize).squeeze(-1).squeeze(-1) # b x 512
        else:
            encoded_feat = images # in case of args.load_feat input images are pretrained resnet18 features of b x 512 size

        # call decoder
        if args.type == "vox":
            # TODO:
            # voxels_pred = 
            x=self.fc(encoded_feat)
            x=x.view(B,128,4,4,4)
            voxels_pred = self.decoder(x)
            if not self.training:
                voxels_pred = torch.sigmoid(voxels_pred)
            return voxels_pred

        elif args.type == "point":
            # TODO:
            # pointclouds_pred =     
            pointclouds_pred = self.decoder(encoded_feat)             # b x (n_points*3)
            pointclouds_pred = pointclouds_pred.view(B, self.n_point, 3)
            return pointclouds_pred        
            

        elif args.type == "mesh":
            # TODO:
            deform_vertices_pred =  self.decoder(encoded_feat)           
            mesh_pred = self.mesh_pred.offset_verts(deform_vertices_pred.reshape([-1,3]))
            return  mesh_pred 
        elif args.type == "implicit":
            if self.training:
                # a random subset of grid points per step, to save memory
                idx = torch.randint(0, self.grid.shape[0], (self.n_train_queries,), device=self.grid.device)
            else:
                idx = torch.arange(self.grid.shape[0], device=self.grid.device)   # all 32768 points
            queries = self.grid[idx].unsqueeze(0).expand(B, -1, -1)              # B x Q x 3
            feat = encoded_feat.unsqueeze(1).expand(-1, queries.shape[1], -1)    # B x Q x 512
            logits = self.decoder(torch.cat([feat, queries], dim=-1)).squeeze(-1)   # B x Q
            if self.training:
                return logits, idx          # the loss needs to know which cells were asked
            return torch.sigmoid(logits).view(B, 1, 32, 32, 32)
        elif args.type == "parametric":
            # a fresh random set of 2D samples in the unit square for every shape
            uv = torch.rand(B, self.n_point, 2, device=encoded_feat.device)       # B x N x 2
            feat = encoded_feat.unsqueeze(1).expand(-1, self.n_point, -1)        # B x N x 512
            points = self.decoder(torch.cat([feat, uv], dim=-1))                 # B x N x 3
            return points

                

