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
from vista.model.equi.vista_tactile_to_sphere import VistaTactileToSphere
from vista.model.equi.vista_util import S2Conv, SO3Conv, s2_healpix_grid, so3_near_identity_grid


class SphereTokenConcateFusion(nn.Module):
    """Naively concatenate visual and tactile sphere tokens, then project."""

    def __init__(
        self,
        visual_dim: int,
        tactile_dim: int,
        out_dim: int,
        num_tactile_sources: int,
    ):
        super().__init__()
        in_dim = int(visual_dim) + int(tactile_dim) * int(num_tactile_sources)
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, int(out_dim)),
            nn.ReLU(inplace=True),
        )

    def forward(
        self,
        visual_tokens: torch.Tensor,
        tactile_tokens: Sequence[torch.Tensor],
    ) -> torch.Tensor:
        return self.net(torch.cat([visual_tokens, *tuple(tactile_tokens)], dim=-1))


class VistaConcateSpherePolicy(nn.Module):
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
        attention_heads: int = 8,
        attention_head_dim: int = 64,
        tactile_mode: str = "raw",
    ):
        super().__init__()
        self.lmax = int(lmax)
        self.tactile_dim = int(tactile_dim)
        if str(tactile_mode) != "raw":
            raise ValueError(
                "VISTA concate supports only tactile_mode='raw' "
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
        self.image_to_sphere = VistaImageToSphere(
            fmap_shape=self.image_encoder.output_shape,
            lmax=lmax,
            rec_level=rec_level,
            max_beta=max_beta,
        )

        self._single_left_tactile = self.tactile_sides == ("left",)
        if self._single_left_tactile:
            self.tactile_to_sphere = VistaTactileToSphere(
                tactile_shape=tactile_shape,
                encoder=encoder,
                feature_dim=visual_dim,
                token_dim=tactile_dim,
                sphere_dirs=self.image_to_sphere.sphere_dirs,
                harmonic_Y=self.image_to_sphere.Y,
                harmonic_omega=self.image_to_sphere.omega,
                N=N,
                initialize=initialize,
            )
        else:
            tactile_to_sphere = nn.ModuleDict()
            for side in self.tactile_sides:
                side_shape = (
                    right_tactile_shape
                    if side == "right" and right_tactile_shape is not None
                    else tactile_shape
                )
                tactile_to_sphere[side] = VistaTactileToSphere(
                    tactile_shape=side_shape,
                    encoder=encoder,
                    feature_dim=visual_dim,
                    token_dim=tactile_dim,
                    sphere_dirs=self.image_to_sphere.sphere_dirs,
                    harmonic_Y=self.image_to_sphere.Y,
                    harmonic_omega=self.image_to_sphere.omega,
                    N=N,
                    initialize=initialize,
                )
            self.tactile_to_sphere = tactile_to_sphere

        self.token_fusion = SphereTokenConcateFusion(
            visual_dim=visual_dim,
            tactile_dim=tactile_dim,
            out_dim=fused_dim,
            num_tactile_sources=len(self.tactile_sides),
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

    def _encode_tactile_side(
        self,
        side: str,
        image: torch.Tensor | None,
        visual_tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, bool]:
        if image is None:
            if not self.allow_missing_tactile:
                raise ValueError(f"Missing required tactile image for side='{side}'")
            n, p, _ = visual_tokens.shape
            return visual_tokens.new_zeros((n, p, self.tactile_dim)), False
        if self._single_left_tactile:
            return self.tactile_to_sphere(image), True
        return self.tactile_to_sphere[side](image), True

    def forward(
        self,
        image: torch.Tensor,
        left_tactile: torch.Tensor | None,
        eef_quat: torch.Tensor,
        right_tactile: torch.Tensor | None = None,
    ) -> torch.Tensor:
        fmap = self._as_tensor(self.image_encoder(image))
        if fmap.shape[-2:] != (7, 7):
            fmap = F.adaptive_avg_pool2d(fmap, (7, 7))
        visual_tokens = self.image_to_sphere.project_tokens(fmap)

        tactile_images = {
            "left": left_tactile,
            "right": right_tactile,
        }
        tactile_tokens_by_side = {}
        tactile_tokens = []
        for side in self.tactile_sides:
            side_tokens, _ = self._encode_tactile_side(
                side,
                tactile_images.get(side),
                visual_tokens,
            )
            tactile_tokens_by_side[side] = side_tokens
            tactile_tokens.append(side_tokens)

        fused_tokens = self.token_fusion(visual_tokens, tactile_tokens)
        coeff = self.image_to_sphere.harmonic_project(fused_tokens)
        coeff = torch.einsum("nij,ncj->nci", self.irreps.D_from_quaternion(eef_quat), coeff)
        x = self.s2_conv(coeff)
        x = self.s2_act(x)
        x = self.so3_conv_1(x)
        x = self.so3_act_1(x)
        out = self.so3_conv_2(x)
        self.last_debug = {
            "fusion": "sphere_concate",
            "visual_tokens": visual_tokens.detach(),
            "tactile_tokens": {
                side: tokens.detach() for side, tokens in tactile_tokens_by_side.items()
            },
            "fused_tokens": fused_tokens.detach(),
            "attention": None,
        }
        return out
