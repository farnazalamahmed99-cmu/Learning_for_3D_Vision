import torch

from typing import List, Optional, Tuple
from pytorch3d.renderer.cameras import CamerasBase
from ray_utils import RayBundle


# Volume renderer which integrates color and density along rays
# according to the equations defined in [Mildenhall et al. 2020]
class VolumeRenderer(torch.nn.Module):
    def __init__(
        self,
        cfg
    ):
        super().__init__()

        self._chunk_size = cfg.chunk_size
        self._white_background = cfg.white_background if 'white_background' in cfg else False

    def _compute_weights(
        self,
        deltas,
        rays_density: torch.Tensor,
        eps: float = 1e-10
    ):
        # TODO (1.5): Compute transmittance using the equation described in the README
        alpha=1.0-torch.exp(-rays_density*deltas)
        T=torch.cumprod(1.0-alpha+eps,dim=1)
        T = torch.cat([torch.ones_like(T[:, :1, :]), T[:, :-1, :]], dim=1)

        # TODO (1.5): Compute weight used for rendering from transmittance and alpha
        weights = alpha * T
        return weights
    
    def _aggregate(
        self,
        weights: torch.Tensor,
        rays_feature: torch.Tensor
    ):
        # TODO (1.5): Aggregate (weighted sum of) features using weights
        feature = torch.sum(weights * rays_feature, dim=1)

        return feature

    def forward(
        self,
        sampler,
        implicit_fn,
        ray_bundle,
    ):
        B = ray_bundle.shape[0]

        # Process the chunks of rays.
        chunk_outputs = []

        for chunk_start in range(0, B, self._chunk_size):
            cur_ray_bundle = ray_bundle[chunk_start:chunk_start+self._chunk_size]

            # Sample points along the ray
            cur_ray_bundle = sampler(cur_ray_bundle)
            n_pts = cur_ray_bundle.sample_shape[1]

            # Call implicit function with sample points
            implicit_output = implicit_fn(cur_ray_bundle)
            density = implicit_output['density']
            feature = implicit_output['feature']

            # Compute length of each ray segment
            depth_values = cur_ray_bundle.sample_lengths[..., 0]
            deltas = torch.cat(
                (
                    depth_values[..., 1:] - depth_values[..., :-1],
                    1e10 * torch.ones_like(depth_values[..., :1]),
                ),
                dim=-1,
            )[..., None]

            # Compute aggregation weights
            weights = self._compute_weights(
                deltas.view(-1, n_pts, 1),
                density.view(-1, n_pts, 1)
            ) 

            # TODO (1.5): Render (color) features using weights
            feature=self._aggregate(weights, feature.view(-1, n_pts, feature.shape[-1]))

            # TODO (1.5): Render depth map
            depth = self._aggregate(weights, depth_values.view(-1, n_pts, 1))

            # Return
            cur_out = {
                'feature': feature,
                'depth': depth,
            }

            chunk_outputs.append(cur_out)

        # Concatenate chunk outputs
        out = {
            k: torch.cat(
              [chunk_out[k] for chunk_out in chunk_outputs],
              dim=0
            ) for k in chunk_outputs[0].keys()
        }

        return out


# Q4.2: Draw n_samples depths per ray from the piecewise-constant PDF defined by
# the coarse weights (inverse transform sampling, Mildenhall et al. 2020, Sec 5.2)
def sample_pdf(bins, weights, n_samples, deterministic=False):
    # bins: (N, M+1) bin edges, weights: (N, M)
    weights = weights + 1e-5
    pdf = weights / torch.sum(weights, dim=-1, keepdim=True)
    cdf = torch.cumsum(pdf, dim=-1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], dim=-1)  # (N, M+1)

    if deterministic:
        u = torch.linspace(0.0, 1.0, n_samples, device=bins.device)
        u = u.expand(cdf.shape[0], n_samples).contiguous()
    else:
        u = torch.rand(cdf.shape[0], n_samples, device=bins.device)

    # Find the CDF interval each u falls in, then invert linearly inside it
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.clamp(inds - 1, min=0)
    above = torch.clamp(inds, max=cdf.shape[-1] - 1)

    cdf_below = torch.gather(cdf, -1, below)
    cdf_above = torch.gather(cdf, -1, above)
    bins_below = torch.gather(bins, -1, below)
    bins_above = torch.gather(bins, -1, above)

    denom = cdf_above - cdf_below
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_below) / denom

    return bins_below + t * (bins_above - bins_below)


# Q4.2: Coarse/fine volume renderer. The coarse network is evaluated at the
# stratified samples, its weights guide where extra samples are placed, and the
# fine network is evaluated at the union of both sets of samples.
class HierarchicalVolumeRenderer(VolumeRenderer):
    def __init__(
        self,
        cfg
    ):
        super().__init__(cfg)

        self._n_pts_fine = cfg.n_pts_fine

    def _render(self, implicit_fn, ray_bundle):
        n_pts = ray_bundle.sample_shape[1]

        implicit_output = implicit_fn(ray_bundle)
        density = implicit_output['density']
        feature = implicit_output['feature']

        depth_values = ray_bundle.sample_lengths[..., 0]
        deltas = torch.cat(
            (
                depth_values[..., 1:] - depth_values[..., :-1],
                1e10 * torch.ones_like(depth_values[..., :1]),
            ),
            dim=-1,
        )[..., None]

        weights = self._compute_weights(
            deltas.view(-1, n_pts, 1),
            density.view(-1, n_pts, 1)
        )

        feature = self._aggregate(weights, feature.view(-1, n_pts, feature.shape[-1]))
        depth = self._aggregate(weights, depth_values.view(-1, n_pts, 1))

        return feature, depth, weights

    def forward(
        self,
        sampler,
        implicit_fn,
        ray_bundle,
    ):
        B = ray_bundle.shape[0]

        # Process the chunks of rays.
        chunk_outputs = []

        for chunk_start in range(0, B, self._chunk_size):
            cur_ray_bundle = ray_bundle[chunk_start:chunk_start+self._chunk_size]

            # Coarse pass at the stratified samples
            cur_ray_bundle = sampler(cur_ray_bundle)
            feature_coarse, _, weights_coarse = self._render(implicit_fn.coarse, cur_ray_bundle)

            # Importance sample new depths where the coarse weights are large.
            # Bins are the midpoints between coarse samples; the weights are
            # detached so the fine loss does not push the coarse network.
            z_coarse = cur_ray_bundle.sample_lengths[..., 0]
            z_mid = 0.5 * (z_coarse[..., 1:] + z_coarse[..., :-1])
            z_fine = sample_pdf(
                z_mid,
                weights_coarse[:, 1:-1, 0].detach(),
                self._n_pts_fine,
                deterministic=not self.training,
            )
            z_all, _ = torch.sort(torch.cat([z_coarse, z_fine.detach()], dim=-1), dim=-1)

            # Fine pass at the union of coarse and fine samples
            sample_points = (
                cur_ray_bundle.origins[:, None, :]
                + z_all[..., None] * cur_ray_bundle.directions[:, None, :]
            )
            fine_ray_bundle = RayBundle(
                cur_ray_bundle.origins,
                cur_ray_bundle.directions,
                sample_points,
                z_all[..., None],
            )
            feature, depth, _ = self._render(implicit_fn.fine, fine_ray_bundle)

            # Return
            cur_out = {
                'feature': feature,
                'depth': depth,
                'feature_coarse': feature_coarse,
            }

            chunk_outputs.append(cur_out)

        # Concatenate chunk outputs
        out = {
            k: torch.cat(
              [chunk_out[k] for chunk_out in chunk_outputs],
              dim=0
            ) for k in chunk_outputs[0].keys()
        }

        return out


def sdf_to_density(signed_distance, alpha, beta):
    # TODO (Q4.3): Convert signed distance to density with alpha, beta parameters
    # VolSDF Eq. 2-3: sigma = alpha * Psi_beta(-d), Psi_beta = CDF of a zero-mean Laplace with scale beta
    # exp(-|d|/beta) is used in both branches so neither can overflow (which would give NaN gradients)
    half_exp = 0.5 * torch.exp(-torch.abs(signed_distance) / beta)
    return alpha * torch.where(signed_distance > 0, half_exp, 1.0 - half_exp)

class VolumeSDFRenderer(VolumeRenderer):
    def __init__(
        self,
        cfg
    ):
        super().__init__(cfg)

        self._chunk_size = cfg.chunk_size
        self._white_background = cfg.white_background if 'white_background' in cfg else False
        self.alpha = cfg.alpha
        self.beta = cfg.beta

        self.cfg = cfg

    def forward(
        self,
        sampler,
        implicit_fn,
        ray_bundle,
        light_dir=None
    ):
        B = ray_bundle.shape[0]

        # Process the chunks of rays.
        chunk_outputs = []

        for chunk_start in range(0, B, self._chunk_size):
            cur_ray_bundle = ray_bundle[chunk_start:chunk_start+self._chunk_size]

            # Sample points along the ray
            cur_ray_bundle = sampler(cur_ray_bundle)
            n_pts = cur_ray_bundle.sample_shape[1]

            # Call implicit function with sample points
            distance, color = implicit_fn.get_distance_color(cur_ray_bundle.sample_points)
            density = sdf_to_density(distance, self.alpha, self.beta) # TODO (Q4.3): convert SDF to density

            # Compute length of each ray segment
            depth_values = cur_ray_bundle.sample_lengths[..., 0]
            deltas = torch.cat(
                (
                    depth_values[..., 1:] - depth_values[..., :-1],
                    1e10 * torch.ones_like(depth_values[..., :1]),
                ),
                dim=-1,
            )[..., None]

            # Compute aggregation weights
            weights = self._compute_weights(
                deltas.view(-1, n_pts, 1),
                density.view(-1, n_pts, 1)
            ) 

            geometry_color = torch.zeros_like(color)

            # Compute color
            color = self._aggregate(
                weights,
                color.view(-1, n_pts, color.shape[-1])
            )

            # Return
            cur_out = {
                'feature': color,
                "geometry": geometry_color
            }

            chunk_outputs.append(cur_out)

        # Concatenate chunk outputs
        out = {
            k: torch.cat(
              [chunk_out[k] for chunk_out in chunk_outputs],
              dim=0
            ) for k in chunk_outputs[0].keys()
        }

        return out


renderer_dict = {
    'volume': VolumeRenderer,
    'volume_hierarchical': HierarchicalVolumeRenderer,
    'volume_sdf': VolumeSDFRenderer
}
