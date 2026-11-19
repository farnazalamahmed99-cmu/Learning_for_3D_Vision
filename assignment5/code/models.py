import torch
import torch.nn as nn
import torch.nn.functional as F

# ------ TO DO ------
class cls_model(nn.Module):
    def __init__(self, num_classes=3):
        super(cls_model, self).__init__()
        self.point_mlp=nn.Sequential(
            nn.Conv1d(3, 64, 1),nn.BatchNorm1d(64),nn.ReLU(),
            nn.Conv1d(64, 64, 1),nn.BatchNorm1d(64),nn.ReLU(),
            nn.Conv1d(64, 64, 1),nn.BatchNorm1d(64),nn.ReLU(),
            nn.Conv1d(64, 128, 1),nn.BatchNorm1d(128),nn.ReLU(),
            nn.Conv1d(128, 1024, 1),nn.BatchNorm1d(1024),nn.ReLU(),
        )
       
        self.classifier=nn.Sequential(
            nn.Linear(1024, 512),nn.BatchNorm1d(512),nn.ReLU(),
            nn.Linear(512, 256),nn.BatchNorm1d(256),nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, points):
        '''
        points: tensor of size (B, N, 3)
                , where B is batch size and N is the number of points per object (N=10000 by default)
        output: tensor of size (B, num_classes)
        '''
        x=points.transpose(1,2)
        x=self.point_mlp(x)
        x=torch.max(x,dim=2)[0]
        x=self.classifier(x)
        return x



# ------ TO DO ------
class seg_model(nn.Module):
    def __init__(self, num_seg_classes = 6):
        super(seg_model, self).__init__()
        self.local_mlp=nn.Sequential(
            nn.Conv1d(3,64,1),nn.BatchNorm1d(64),nn.ReLU(),
            nn.Conv1d(64,64,1),nn.BatchNorm1d(64),nn.ReLU(),
            
        )
        self.global_mlp=nn.Sequential(
            nn.Conv1d(64, 64, 1),nn.BatchNorm1d(64),nn.ReLU(),
            nn.Conv1d(64, 128, 1),nn.BatchNorm1d(128),nn.ReLU(),
            nn.Conv1d(128, 1024, 1),nn.BatchNorm1d(1024),nn.ReLU(),
        )
        self.seg_head=nn.Sequential(
            nn.Conv1d(1088, 512, 1),nn.BatchNorm1d(512),nn.ReLU(),
            nn.Conv1d(512, 256, 1),nn.BatchNorm1d(256),nn.ReLU(),
            nn.Conv1d(256, 128, 1),nn.BatchNorm1d(128),nn.ReLU(),
            nn.Conv1d(128, num_seg_classes, 1)
        )

    def forward(self, points):
        '''
        points: tensor of size (B, N, 3)
                , where B is batch size and N is the number of points per object (N=10000 by default)
        output: tensor of size (B, N, num_seg_classes)
        '''
        N=points.size(1)
        x=points.transpose(1,2)
        local=self.local_mlp(x)
        g=self.global_mlp(local)
        g=torch.max(g, dim=2, keepdim=True)[0]
        g=g.repeat(1,1,N)
        x=torch.cat([local,g],dim=1)
        x=self.seg_head(x)
        x=x.transpose(1,2)
        return x


# ------ Q4: DGCNN (EdgeConv) - adds locality via k-nearest-neighbour graphs ------
def knn(x, k):
    """x: (B, C, N) -> indices (B, N, k) of each point's k nearest neighbours (in C-dim space)."""
    inner = -2 * torch.matmul(x.transpose(2, 1), x)          # (B, N, N)   -2 xi.xj
    xx = (x ** 2).sum(dim=1, keepdim=True)                   # (B, 1, N)   ||xj||^2
    neg_dist = -xx.transpose(2, 1) - inner - xx              # -||xi - xj||^2
    return neg_dist.topk(k=k, dim=-1)[1]                     # largest = closest (includes itself)


def edge_features(x, k):
    """x: (B, C, N) -> EdgeConv input (B, 2C, N, k): [x_i, x_j - x_i] for each neighbour j of i."""
    B, C, N = x.shape
    idx = knn(x, k)                                          # (B, N, k)
    pts = x.transpose(1, 2).contiguous()                     # (B, N, C)
    base = torch.arange(B, device=x.device).view(-1, 1, 1) * N
    nbrs = pts.view(B * N, C)[(idx + base).view(-1)].view(B, N, k, C)   # gather neighbour features
    center = pts.unsqueeze(2).expand(-1, -1, k, -1)          # (B, N, k, C)
    feat = torch.cat([center, nbrs - center], dim=3)         # (B, N, k, 2C)
    return feat.permute(0, 3, 1, 2).contiguous()             # (B, 2C, N, k) for Conv2d


def edgeconv(c_in, c_out):
    """Shared MLP on every (point, neighbour) edge. Conv2d 1x1 = Linear on each edge."""
    return nn.Sequential(nn.Conv2d(2 * c_in, c_out, 1, bias=False), nn.BatchNorm2d(c_out), nn.LeakyReLU(0.2))


class dgcnn_cls(nn.Module):
    def __init__(self, num_classes=3, k=20):
        super(dgcnn_cls, self).__init__()
        self.k = k
        self.ecs = nn.ModuleList([edgeconv(3, 64), edgeconv(64, 64), edgeconv(64, 128), edgeconv(128, 256)])
        self.fuse = nn.Sequential(nn.Conv1d(512, 1024, 1, bias=False), nn.BatchNorm1d(1024), nn.LeakyReLU(0.2))
        self.head = nn.Sequential(
            nn.Linear(1024, 512, bias=False), nn.BatchNorm1d(512), nn.LeakyReLU(0.2), nn.Dropout(0.5),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.LeakyReLU(0.2), nn.Dropout(0.5),
            nn.Linear(256, num_classes))

    def forward(self, points):                               # (B, N, 3)
        x = points.transpose(1, 2)                           # (B, 3, N)
        feats = []
        for ec in self.ecs:
            x = ec(edge_features(x, self.k)).max(dim=3)[0]   # max over k neighbours -> (B, C_out, N)
            feats.append(x)                                  # graph is rebuilt in feature space next layer ("dynamic")
        x = self.fuse(torch.cat(feats, dim=1))               # (B, 1024, N)
        x = x.max(dim=2)[0]                                  # global max over points -> (B, 1024)
        return self.head(x)                                  # (B, num_classes)


class dgcnn_seg(nn.Module):
    def __init__(self, num_seg_classes=6, k=20):
        super(dgcnn_seg, self).__init__()
        self.k = k
        self.ecs = nn.ModuleList([edgeconv(3, 64), edgeconv(64, 64), edgeconv(64, 64)])
        self.fuse = nn.Sequential(nn.Conv1d(192, 1024, 1, bias=False), nn.BatchNorm1d(1024), nn.LeakyReLU(0.2))
        self.head = nn.Sequential(
            nn.Conv1d(192 + 1024, 256, 1, bias=False), nn.BatchNorm1d(256), nn.LeakyReLU(0.2),
            nn.Conv1d(256, 256, 1, bias=False), nn.BatchNorm1d(256), nn.LeakyReLU(0.2), nn.Dropout(0.5),
            nn.Conv1d(256, 128, 1, bias=False), nn.BatchNorm1d(128), nn.LeakyReLU(0.2),
            nn.Conv1d(128, num_seg_classes, 1))

    def forward(self, points):                               # (B, N, 3)
        N = points.size(1)
        x = points.transpose(1, 2)                           # (B, 3, N)
        feats = []
        for ec in self.ecs:
            x = ec(edge_features(x, self.k)).max(dim=3)[0]   # (B, 64, N)
            feats.append(x)
        local = torch.cat(feats, dim=1)                      # (B, 192, N)   multi-scale local features
        g = self.fuse(local).max(dim=2, keepdim=True)[0]     # (B, 1024, 1)  global feature
        g = g.expand(-1, -1, N)                              # (B, 1024, N)
        x = self.head(torch.cat([local, g], dim=1))          # (B, 6, N)
        return x.transpose(1, 2)                             # (B, N, 6)



