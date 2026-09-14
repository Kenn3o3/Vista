from __future__ import annotations

import math
from typing import Sequence

import e3nn
import torch
import torch.nn as nn
import torch.nn.functional as F
from e3nn import o3

from vista.model.equi.i2s_policy import ImageEncoder
from vista.model.equi.vista_image_to_sphere import VistaImageToSphere
from vista.model.equi.vista_util import S2Conv, SO3Conv, s2_healpix_grid, so3_near_identity_grid


def _valid_num_groups(num_groups: int, num_channels: int) -> int:
    groups = min(int(num_groups), int(num_channels))
    while groups > 1 and num_channels % groups != 0:
        groups -= 1
    return groups


class FeatureMapFusion(nn.Module):
    """Fuse visual and tactile 2D feature maps before the sphere lift."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_groups: int = 32,
    ):
        super().__init__()
        groups = _valid_num_groups(num_groups, out_channels)
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, feature_maps: Sequence[torch.Tensor]) -> torch.Tensor:
        return self.net(torch.cat(tuple(feature_maps), dim=1))


class VistaEarlyFusionSpherePolicy(nn.Module):
    def __init__(
        self,
        encoder: str = "equiresnet50",
        lmax: int = 6,
        visual_dim: int = 1024,
        tactile_dim: int = 1024,
        fused_dim: int = 1024,
        so3_dim: int = 64,
        N: int = 8,
        initialize: bool = True,
        f_out: int = 128,
        tactile_shape=(3, 84, 84),
        right_tactile_shape=None,
        tactile_sides: Sequence[str] = ("left",),
        allow_missing_tactile: bool = False,
        rec_level: int = 3,
        max_beta: float = math.pi / 2,
        tactile_mode: str = "raw",
        fusion_groups: int = 32,
    ):
        super().__init__()
        self.lmax = int(lmax)
        self.tactile_dim = int(tactile_dim)
        if str(tactile_mode) != "raw":
            raise ValueError(
                "VISTA early-fusion supports only tactile_mode='raw' "
                "(tactile RGB image with markers)."
            )
        if isinstance(tactile_sides, str):
            tactile_sides = (tactile_sides,)
        self.tactile_sides = tuple(tactile_sides)
        if not self.tactile_sides:
            raise ValueError("tactile_sides must contain at least one side")
        for side in self.tactile_sides:
            if side not in {"left", "right"}:
                raise ValueError(f"Unsupported tactile side: {side}")
        self.allow_missing_tactile = bool(allow_missing_tactile)

        self.image_encoder = ImageEncoder(
            encoder=encoder,
            out_fdim=visual_dim,
            out_shape=(7, 7),
            N=N,
            initialize=initialize,
        )

        self._single_left_tactile = self.tactile_sides == ("left",)
        if self._single_left_tactile:
            self.tactile_encoder = ImageEncoder(
                encoder=encoder,
                out_fdim=tactile_dim,
                out_shape=(7, 7),
                N=N,
                obs_channel=int(tactile_shape[0]),
                initialize=initialize,
            )
        else:
            tactile_encoders = nn.ModuleDict()
            for side in self.tactile_sides:
                side_shape = (
                    right_tactile_shape
                    if side == "right" and right_tactile_shape is not None
                    else tactile_shape
                )
                tactile_encoders[side] = ImageEncoder(
                    encoder=encoder,
                    out_fdim=tactile_dim,
                    out_shape=(7, 7),
                    N=N,
                    obs_channel=int(side_shape[0]),
                    initialize=initialize,
                )
            self.tactile_encoders = tactile_encoders

        self.feature_fusion = FeatureMapFusion(
            in_channels=int(visual_dim) + int(tactile_dim) * len(self.tactile_sides),
            out_channels=int(fused_dim),
            num_groups=fusion_groups,
        )
        self.image_to_sphere = VistaImageToSphere(
            fmap_shape=(int(fused_dim), 7, 7),
            lmax=lmax,
            rec_level=rec_level,
            max_beta=max_beta,
        )

        s2_kernel_grid = s2_healpix_grid(max_beta=math.inf, rec_level=1)
        so3_kernel_grid = so3_near_identity_grid()
        self.irreps = o3.Irreps([(1, (l, 1)) for l in range(lmax + 1)])
        self.s2_conv = S2Conv(fused_dim, so3_dim, lmax, s2_kernel_grid)
        self.s2_act = e3nn.nn.SO3Activation(lmax, lmax, act=torch.relu_, resolution=8)
        self.so3_conv_1 = SO3Conv(so3_dim, 64, lmax, so3_kernel_grid, lmax)
        self.so3_act_1 = e3nn.nn.SO3Activation(lmax, 6, act=torch.relu_, resolution=8)
        self.so3_conv_2 = SO3Conv(64, f_out, 6, so3_kernel_grid, 6)
        self.last_debug = None

    @staticmethod
    def _as_tensor(fmap):
        return fmap if isinstance(fmap, torch.Tensor) else fmap.tensor

    @staticmethod
    def _pool_to_projector_shape(fmap: torch.Tensor) -> torch.Tensor:
        if fmap.shape[-2:] != (7, 7):
            fmap = F.adaptive_avg_pool2d(fmap, (7, 7))
        return fmap

    def _encode_tactile_side(
        self,
        side: str,
        image: torch.Tensor | None,
        ref_fmap: torch.Tensor,
    ) -> tuple[torch.Tensor, bool]:
        if image is None:
            if not self.allow_missing_tactile:
                raise ValueError(f"Missing required tactile image for side='{side}'")
            n, _, h, w = ref_fmap.shape
            return ref_fmap.new_zeros((n, self.tactile_dim, h, w)), False
        if self._single_left_tactile:
            fmap = self._as_tensor(self.tactile_encoder(image))
        else:
            fmap = self._as_tensor(self.tactile_encoders[side](image))
        return self._pool_to_projector_shape(fmap), True

    def forward(
        self,
        image: torch.Tensor,
        left_tactile: torch.Tensor | None,
        eef_quat: torch.Tensor,
        right_tactile: torch.Tensor | None = None,
    ) -> torch.Tensor:
        visual_fmap = self._as_tensor(self.image_encoder(image))
        visual_fmap = self._pool_to_projector_shape(visual_fmap)

        tactile_images = {
            "left": left_tactile,
            "right": right_tactile,
        }
        tactile_fmaps_by_side = {}
        tactile_fmaps = []
        for side in self.tactile_sides:
            side_fmap, _ = self._encode_tactile_side(
                side,
                tactile_images.get(side),
                visual_fmap,
            )
            tactile_fmaps_by_side[side] = side_fmap
            tactile_fmaps.append(side_fmap)

        fused_fmap = self.feature_fusion([visual_fmap, *tactile_fmaps])
        fused_tokens = self.image_to_sphere.project_tokens(fused_fmap)
        coeff = self.image_to_sphere.harmonic_project(fused_tokens)
        coeff = torch.einsum("nij,ncj->nci", self.irreps.D_from_quaternion(eef_quat), coeff)
        x = self.s2_conv(coeff)
        x = self.s2_act(x)
        x = self.so3_conv_1(x)
        x = self.so3_act_1(x)
        out = self.so3_conv_2(x)
        self.last_debug = {
            "fusion": "early_feature_map",
            "visual_fmap": visual_fmap.detach(),
            "tactile_fmaps": {
                side: fmap.detach() for side, fmap in tactile_fmaps_by_side.items()
            },
            "fused_fmap": fused_fmap.detach(),
            "fused_tokens": fused_tokens.detach(),
            "attention": None,
        }
        return out
