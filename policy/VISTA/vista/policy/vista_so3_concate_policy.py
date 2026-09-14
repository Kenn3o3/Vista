from __future__ import annotations

import torch
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from vista.model.common.normalizer import LinearNormalizer
from vista.model.common.rotation_transformer import RotationTransformer
from vista.model.diffusion.mask_generator import LowdimMaskGenerator
from vista.model.equi.equi_conditional_unet1d import EquiDiffusionUNet
from vista.model.equi.equi_group_sampling import EquiGroupSamplingIco
from vista.model.equi.vista_concate_obs_encoder import VISTAConcateObsEncoder
from vista.model.vision.rot_randomizer import RotRandomizer, RotRandomizerForPrediction
from vista.policy.base_image_policy import BaseImagePolicy
from vista.policy.vista_so3_policy import VISTAPolicy as _SO3VISTAPolicy


class VISTAPolicy(_SO3VISTAPolicy):
    def __init__(
        self,
        shape_meta: dict,
        noise_scheduler: DDPMScheduler,
        horizon,
        n_action_steps,
        n_obs_steps,
        num_inference_steps=None,
        crop_shape=(76, 76),
        N=60,
        enc_n_hidden=128,
        diffusion_step_embed_dim=256,
        down_dims=(256, 512, 1024),
        kernel_size=5,
        n_groups=8,
        cond_predict_scale=True,
        rot_aug=False,
        lmax=6,
        visual_dim=1024,
        tactile_dim=1024,
        fused_dim=1024,
        so3_dim=64,
        initialize=True,
        encoder="equiresnet50",
        tactile_shape=(3, 84, 84),
        right_tactile_shape=None,
        tactile_sides=None,
        allow_missing_tactile=False,
        rec_level=3,
        max_beta=1.5707963267948966,
        attention_heads=8,
        attention_head_dim=64,
        tactile_mode="raw",
        **kwargs,
    ):
        BaseImagePolicy.__init__(self)

        action_shape = shape_meta["action"]["shape"]
        assert len(action_shape) == 1
        action_dim = action_shape[0]
        obs_shape_meta = shape_meta["obs"]

        if tactile_sides is None:
            if "robot0_tactile_right_image" in obs_shape_meta:
                tactile_sides = ("left", "right")
            else:
                tactile_sides = ("left",)

        left_tactile_shape = obs_shape_meta.get(
            "robot0_tactile_left_image",
            {"shape": tactile_shape},
        )["shape"]
        if "robot0_tactile_right_image" in obs_shape_meta:
            right_tactile_shape = obs_shape_meta["robot0_tactile_right_image"]["shape"]
        elif right_tactile_shape is None:
            right_tactile_shape = left_tactile_shape

        self.enc = VISTAConcateObsEncoder(
            obs_shape=obs_shape_meta["robot0_eye_in_hand_image"]["shape"],
            crop_shape=crop_shape,
            n_hidden=enc_n_hidden,
            N=8,
            initialize=initialize,
            lmax=lmax,
            visual_dim=visual_dim,
            tactile_dim=tactile_dim,
            fused_dim=fused_dim,
            so3_dim=so3_dim,
            encoder=encoder,
            tactile_shape=left_tactile_shape,
            right_tactile_shape=right_tactile_shape,
            tactile_sides=tactile_sides,
            allow_missing_tactile=allow_missing_tactile,
            rec_level=rec_level,
            max_beta=max_beta,
            attention_heads=attention_heads,
            attention_head_dim=attention_head_dim,
            tactile_mode=tactile_mode,
        )

        obs_feature_dim = enc_n_hidden
        global_cond_dim = obs_feature_dim * n_obs_steps

        self.equi_sampler = EquiGroupSamplingIco(lmax=lmax, f_out=obs_feature_dim)
        self.diff = EquiDiffusionUNet(
            act_emb_dim=64,
            local_cond_dim=None,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            cond_predict_scale=cond_predict_scale,
            N=N,
            lmax=lmax,
        )

        print("Enc params: %e" % sum(p.numel() for p in self.enc.parameters()))
        print(
            "Equi sampler params: %e"
            % sum(p.numel() for p in self.equi_sampler.parameters())
        )
        print("Diff params: %e" % sum(p.numel() for p in self.diff.parameters()))

        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False,
        )
        self.normalizer = LinearNormalizer()
        self.rot_randomizer = RotRandomizer()
        self.rot_randomizer2 = RotRandomizerForPrediction()

        self.horizon = horizon
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.crop_shape = crop_shape
        self.obs_feature_dim = obs_feature_dim
        self.rot_aug = rot_aug

        print("Data Augmentation: ", self.rot_aug)
        print("n_obs_steps: ", n_obs_steps)

        self.kwargs = kwargs
        self.noise_scheduler = noise_scheduler
        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps

        self.register_buffer(
            "ws_center", torch.tensor([0, 0, 0.8], dtype=torch.float32)
        )
        self.sixd2mat = RotationTransformer("rotation_6d", "matrix")
