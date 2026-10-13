import torch
import torch.nn as nn
from models.unet_2d_condition import UNet2DConditionModel
from models.unet_3d import UNet3DConditionModel
from models.pose_guider import PoseGuider

class VFIPoseModel(nn.Module):
    def __init__(
        self,
        reference_unet: UNet2DConditionModel,
        denoising_unet: UNet3DConditionModel,
        seed_guider: PoseGuider,
        pose_guider: PoseGuider,
        reference_control_writer,
        reference_control_reader,
    ):
        super().__init__()
        self.reference_unet = reference_unet
        self.denoising_unet = denoising_unet
        self.pose_guider = pose_guider
        self.seed_guider = seed_guider
        self.reference_control_writer = reference_control_writer
        self.reference_control_reader = reference_control_reader

    def forward(
        self,
        noisy_latents,
        timesteps,
        ref_image_latents,
        clip_image_embeds,
        pose_img,
        seed_img,
        uncond_fwd: bool = False,
    ):
        seed_cond_tensor = seed_img.to(device="cuda")
        seed_fea = self.seed_guider(seed_cond_tensor)

        pose_cond_tensor = pose_img.to(device="cuda")
        pose_fea = self.pose_guider(pose_cond_tensor)

        if not uncond_fwd:
            ref_timesteps = torch.zeros_like(timesteps)
            # bs, c, f h, w = ref_image_latents.shape
            # ref_image_latents = ref_image_latents.reshape()
            self.reference_unet(
                ref_image_latents,
                ref_timesteps,
                encoder_hidden_states=clip_image_embeds,
                return_dict=False,
            )
            self.reference_control_reader.update(self.reference_control_writer)

        pose_fea = seed_fea + pose_fea

        model_pred = self.denoising_unet(
            noisy_latents,
            timesteps,
            guidance_fea=pose_fea,
            encoder_hidden_states=clip_image_embeds,
        ).sample

        return model_pred



class VFIShareVAEModel(nn.Module):
    def __init__(
        self,
        reference_unet: UNet2DConditionModel,
        denoising_unet: UNet3DConditionModel,
        seed_guider,
        pose_guider: PoseGuider,
        reference_control_writer,
        reference_control_reader,
    ):
        super().__init__()
        self.reference_unet = reference_unet
        self.denoising_unet = denoising_unet
        self.pose_guider = pose_guider
        self.seed_guider = seed_guider
        self.reference_control_writer = reference_control_writer
        self.reference_control_reader = reference_control_reader

    def forward(
        self,
        noisy_latents,
        timesteps,
        ref_image_latents,
        clip_image_embeds,
        seed_img = None,
        pose_img = None,
        uncond_fwd: bool = False,
    ):
        # print(seed_img.shape)
        # print(pose_img.shape, seed_img.shape)
        if seed_img is not None:
            seed_cond_tensor = seed_img.to(device="cuda")
            seed_fea = self.seed_guider(seed_cond_tensor)
        else:
            seed_fea = None

        if pose_img is not None:
            pose_cond_tensor = pose_img.to(device="cuda")
            pose_fea = self.pose_guider(pose_cond_tensor)
            if seed_fea is None:
                pose_fea = pose_fea
            else:
                pose_fea = seed_fea + pose_fea
        else:
            pose_fea = seed_fea

        if True:
            ref_timesteps = torch.zeros_like(timesteps)
            self.reference_unet(
                ref_image_latents,
                ref_timesteps,
                encoder_hidden_states=clip_image_embeds,
                return_dict=False,
            )
            self.reference_control_reader.update(self.reference_control_writer)

        model_pred = self.denoising_unet(
            noisy_latents,
            timesteps,
            guidance_fea=pose_fea,
            encoder_hidden_states=clip_image_embeds,
        ).sample

        return model_pred


class VFIShareVAEModel_s1(nn.Module):
    def __init__(
        self,
        reference_unet: UNet2DConditionModel,
        denoising_unet: UNet3DConditionModel,
        seed_guider,
        pose_guider: PoseGuider,
        reference_control_writer,
        reference_control_reader,
    ):
        super().__init__()
        self.reference_unet = reference_unet
        self.denoising_unet = denoising_unet
        self.pose_guider = pose_guider
        self.seed_guider = seed_guider
        self.reference_control_writer = reference_control_writer
        self.reference_control_reader = reference_control_reader

    def forward(
        self,
        noisy_latents,
        timesteps,
        ref_image_latents,
        clip_image_embeds,
        seed_img = None,
        pose_img = None,
        uncond_fwd: bool = False,
    ):
        # print(seed_img.shape)
        # print(pose_img.shape, seed_img.shape)
        if seed_img is not None:
            seed_cond_tensor = seed_img.to(device="cuda")
            seed_fea = self.seed_guider(seed_cond_tensor)
        else:
            seed_fea = None

        if pose_img is not None:
            pose_cond_tensor = pose_img.to(device="cuda")
            pose_fea = self.pose_guider(pose_cond_tensor)
            if seed_fea is None:
                pose_fea = pose_fea
            else:
                pose_fea = seed_fea + pose_fea
        else:
            pose_fea = seed_fea

        if not uncond_fwd:
            ref_timesteps = torch.zeros_like(timesteps)
            self.reference_unet(
                ref_image_latents,
                ref_timesteps,
                encoder_hidden_states=clip_image_embeds,
                return_dict=False,
            )
            self.reference_control_reader.update(self.reference_control_writer)

        model_pred = self.denoising_unet(
            noisy_latents,
            timesteps,
            guidance_fea=pose_fea,
            encoder_hidden_states=clip_image_embeds,
        ).sample

        return model_pred


class VFINoCFGModel(nn.Module):
    def __init__(
        self,
        reference_unet: UNet2DConditionModel,
        denoising_unet: UNet3DConditionModel,
        seed_guider,
        pose_guider: PoseGuider,
        reference_control_writer,
        reference_control_reader,
    ):
        super().__init__()
        self.reference_unet = reference_unet
        self.denoising_unet = denoising_unet
        self.pose_guider = pose_guider
        self.seed_guider = seed_guider
        self.reference_control_writer = reference_control_writer
        self.reference_control_reader = reference_control_reader

    def forward(
        self,
        noisy_latents,
        timesteps,
        ref_image_latents,
        clip_image_embeds,
        seed_img = None,
        pose_img = None,
        uncond_fwd: bool = False,
    ):
        # print(seed_img.shape)
        # print(pose_img.shape, seed_img.shape)
        if seed_img is not None:
            seed_cond_tensor = seed_img.to(device="cuda")
            seed_fea = self.seed_guider(seed_cond_tensor)
        else:
            seed_fea = None

        if pose_img is not None:
            pose_cond_tensor = pose_img.to(device="cuda")
            pose_fea = self.pose_guider(pose_cond_tensor)
            if seed_fea is None:
                pose_fea = pose_fea
            else:
                pose_fea = seed_fea + pose_fea
        else:
            pose_fea = seed_fea

        if not uncond_fwd:
            ref_timesteps = torch.zeros_like(timesteps)
            self.reference_unet(
                ref_image_latents,
                ref_timesteps,
                encoder_hidden_states=clip_image_embeds,
                return_dict=False,
            )
            self.reference_control_reader.update(self.reference_control_writer)

        model_pred = self.denoising_unet(
            noisy_latents,
            timesteps,
            guidance_fea=pose_fea,
            encoder_hidden_states=clip_image_embeds,
        ).sample

        return model_pred


class VFINoRefModel(nn.Module):
    def __init__(
        self,
        reference_unet: UNet2DConditionModel,
        denoising_unet: UNet3DConditionModel,
        seed_guider,
        pose_guider: PoseGuider,
        reference_control_writer,
        reference_control_reader,
    ):
        super().__init__()
        self.reference_unet = reference_unet
        self.denoising_unet = denoising_unet
        self.pose_guider = pose_guider
        self.seed_guider = seed_guider
        self.reference_control_writer = reference_control_writer
        self.reference_control_reader = reference_control_reader

    def forward(
        self,
        noisy_latents,
        timesteps,
        ref_image_latents,
        clip_image_embeds,
        seed_img = None,
        pose_img = None,
        uncond_fwd: bool = False,
    ):
        # print(seed_img.shape)
        if seed_img is not None:
            seed_cond_tensor = seed_img.to(device="cuda")
            seed_fea = self.seed_guider(seed_cond_tensor)
        else:
            seed_fea = None

        if pose_img is not None:
            pose_cond_tensor = pose_img.to(device="cuda")
            pose_fea = self.pose_guider(pose_cond_tensor)
            if seed_fea is None:
                pose_fea = pose_fea
            else:
                pose_fea = seed_fea + pose_fea
        else:
            pose_fea = seed_fea

        model_pred = self.denoising_unet(
            noisy_latents,
            timesteps,
            guidance_fea=pose_fea,
            encoder_hidden_states=clip_image_embeds,
        ).sample

        return model_pred


# v1-dup
# reference image are cat to noise, re-train convin
# v2-dup
# reference add to, 2 convin
# v1-blank
# leave middle padding
# v2-blank
# all use 256, batch size 4

# v3-add, flexiable reference
class VFICatModel(nn.Module):
    def __init__(
        self,
        reference_unet: UNet2DConditionModel,
        denoising_unet: UNet3DConditionModel,
        pose_guider: PoseGuider,
        reference_control_writer,
        reference_control_reader,
    ):
        super().__init__()
        self.reference_unet = reference_unet
        self.denoising_unet = denoising_unet
        self.pose_guider = pose_guider
        self.reference_control_writer = reference_control_writer
        self.reference_control_reader = reference_control_reader

    def forward(
        self,
        noisy_latents,
        timesteps,
        ref_image_latents,
        clip_image_embeds,
        seed_img = None,
        pose_img = None,
        uncond_fwd: bool = False,
    ):
        if seed_img is not None:
            seed_cond_tensor = seed_img.to(device="cuda")
            noisy_latents = torch.cat([noisy_latents, seed_cond_tensor], dim=1)
        else:
            noisy_latents = torch.cat([noisy_latents, torch.zeros_like(noisy_latents).to(device="cuda")], dim=1)

        if pose_img is not None:
            pose_cond_tensor = pose_img.to(device="cuda")
            pose_fea = self.pose_guider(pose_cond_tensor)
        else:
            pose_fea = None
        if not uncond_fwd:
            ref_timesteps = torch.zeros_like(timesteps)
            self.reference_unet(
                ref_image_latents,
                ref_timesteps,
                encoder_hidden_states=clip_image_embeds,
                return_dict=False,
            )
            self.reference_control_reader.update(self.reference_control_writer)

        model_pred = self.denoising_unet(
            noisy_latents,
            timesteps,
            guidance_fea=pose_fea,
            encoder_hidden_states=clip_image_embeds,
        ).sample
        return model_pred

from einops import rearrange
class VFIMultiRefModel(nn.Module):
    def __init__(
        self,
        reference_unet: UNet2DConditionModel,
        denoising_unet: UNet3DConditionModel,
        seed_guider,
        pose_guider: PoseGuider,
        reference_control_writer,
        reference_control_reader,
    ):
        super().__init__()
        self.reference_unet = reference_unet
        self.denoising_unet = denoising_unet
        self.pose_guider = pose_guider
        self.seed_guider = seed_guider
        self.reference_control_writer = reference_control_writer
        self.reference_control_reader = reference_control_reader

    def forward(
        self,
        noisy_latents,
        timesteps,
        ref_image_latents,
        clip_image_embeds,
        seed_img = None,
        pose_img = None,
        uncond_fwd: bool = False,
    ):
        if seed_img is not None:
            seed_cond_tensor = seed_img.to(device="cuda")
            seed_fea = self.seed_guider(seed_cond_tensor)
        else:
            seed_fea = None

        if pose_img is not None:
            pose_cond_tensor = pose_img.to(device="cuda")
            pose_fea = self.pose_guider(pose_cond_tensor)
            if seed_fea is None:
                pose_fea = pose_fea
            else:
                pose_fea = seed_fea + pose_fea
        else:
            pose_fea = seed_fea

        if not uncond_fwd:
            # print(timesteps.shape)
            ref_image_latents = rearrange(
                ref_image_latents, "b c f h w -> (b f) c h w"
            )
            ref_timesteps = torch.zeros(ref_image_latents.shape[0]).to(device="cuda")
            # print(ref_timesteps.shape, ref_image_latents.shape, clip_image_embeds.shape)
            self.reference_unet(
                ref_image_latents,
                ref_timesteps,
                encoder_hidden_states=clip_image_embeds,
                return_dict=False,
            )
            self.reference_control_reader.update(self.reference_control_writer)

        model_pred = self.denoising_unet(
            noisy_latents,
            timesteps,
            guidance_fea=pose_fea,
            encoder_hidden_states=clip_image_embeds,
        ).sample

        return model_pred

from einops import rearrange
class VFIMultiRefModel_v4_dup(nn.Module):
    def __init__(
        self,
        reference_unet: UNet2DConditionModel,
        denoising_unet: UNet3DConditionModel,
        seed_guider,
        pose_guider: PoseGuider,
        reference_control_writer,
        reference_control_reader,
    ):
        super().__init__()
        self.reference_unet = reference_unet
        self.denoising_unet = denoising_unet
        self.pose_guider = pose_guider
        self.seed_guider = seed_guider
        self.reference_control_writer = reference_control_writer
        self.reference_control_reader = reference_control_reader

    def forward(
        self,
        noisy_latents,
        timesteps,
        ref_image_latents,
        clip_image_embeds,
        seed_img = None,
        pose_img = None,
        uncond_fwd: bool = False,
    ):
        if seed_img is not None:
            seed_cond_tensor = seed_img.to(device="cuda")
            seed_fea = self.seed_guider(seed_cond_tensor)
        else:
            seed_fea = None

        if pose_img is not None:
            pose_cond_tensor = pose_img.to(device="cuda")
            pose_fea = self.pose_guider(pose_cond_tensor)
            if seed_fea is None:
                pose_fea = pose_fea
            else:
                pose_fea = seed_fea + pose_fea
        else:
            pose_fea = seed_fea

        if not uncond_fwd:
            # print(ref_image_latents.shape)
            ref_image_latents = rearrange(
                ref_image_latents, "b c f h w -> (b f) c h w"
            )
            clip_image_embeds_a = clip_image_embeds.repeat(noisy_latents.shape[2], 1, 1)
            # print(clip_image_embeds.shape, clip_image_embeds_a.shape)
            ref_timesteps = torch.zeros(ref_image_latents.shape[0]).to(device="cuda")
            self.reference_unet(
                ref_image_latents,
                ref_timesteps,
                encoder_hidden_states=clip_image_embeds_a,
                return_dict=False,
            )
            self.reference_control_reader.update(self.reference_control_writer)

        model_pred = self.denoising_unet(
            noisy_latents,
            timesteps,
            guidance_fea=pose_fea,
            encoder_hidden_states=clip_image_embeds,
        ).sample

        return model_pred

class VFIMultiRefModel_v5_dup(nn.Module):
    def __init__(
        self,
        reference_unet: UNet2DConditionModel,
        denoising_unet: UNet3DConditionModel,
        seed_guider,
        pose_guider: PoseGuider,
        reference_control_writer,
        reference_control_reader,
    ):
        super().__init__()
        self.reference_unet = reference_unet
        self.denoising_unet = denoising_unet
        self.pose_guider = pose_guider
        self.seed_guider = seed_guider
        self.reference_control_writer = reference_control_writer
        self.reference_control_reader = reference_control_reader

    def forward(
        self,
        noisy_latents,
        timesteps,
        ref_image_latents,
        clip_image_embeds,
        seed_img = None,
        pose_img = None,
        uncond_fwd: bool = False,
    ):
        if pose_img is not None:
            pose_cond_tensor = pose_img.to(device="cuda")
            pose_fea = self.pose_guider(pose_cond_tensor)
        else:
            pose_fea = None

        # clip_image_embeds_a = rearrange(
        #         clip_image_embeds, "(b a) c f -> b a c f", a=noisy_latents.shape[2],
        #     ) # 2 14 1 768
        # clip_image_embeds_a = clip_image_embeds_a[:, 2]
        # clip_image_embeds_b = clip_image_embeds_a.repeat(1, noisy_latents.shape[2], 1, 1)
        # clip_image_embeds_b = rearrange(
        #         clip_image_embeds_b, "b a c f -> (b a) c f",
        #     )

        if not uncond_fwd:
            ref_image_latents = rearrange(
                ref_image_latents, "b c f h w -> (b f) c h w"
            )
            ref_timesteps = torch.zeros(ref_image_latents.shape[0]).to(device="cuda")
            self.reference_unet(
                ref_image_latents,
                ref_timesteps,
                encoder_hidden_states=clip_image_embeds,
                return_dict=False,
            )
            self.reference_control_reader.update(self.reference_control_writer)

        model_pred = self.denoising_unet(
            noisy_latents,
            timesteps,
            guidance_fea=pose_fea,
            encoder_hidden_states=clip_image_embeds,
        ).sample

        return model_pred

class VFIMultiRefModelv2(nn.Module):
    def __init__(
        self,
        reference_unet: UNet2DConditionModel,
        denoising_unet: UNet3DConditionModel,
        seed_guider,
        pose_guider: PoseGuider,
        reference_control_writer,
        reference_control_reader,
    ):
        super().__init__()
        self.reference_unet = reference_unet
        self.denoising_unet = denoising_unet
        self.pose_guider = pose_guider
        self.seed_guider = seed_guider
        self.reference_control_writer = reference_control_writer
        self.reference_control_reader = reference_control_reader

    def forward(
        self,
        noisy_latents,
        timesteps,
        ref_image_latents,
        clip_image_embeds,
        seed_img = None,
        pose_img = None,
        uncond_fwd: bool = False,
    ):
        if pose_img is not None:
            pose_cond_tensor = pose_img.to(device="cuda")
            pose_fea = self.pose_guider(pose_cond_tensor)
        else:
            pose_fea = None

        if not uncond_fwd:
            # print(timesteps.shape)
            ref_image_latents = rearrange(
                ref_image_latents, "b c f h w -> (b f) c h w"
            )
            ref_timesteps = torch.zeros(ref_image_latents.shape[0]).to(device="cuda")
            print(ref_timesteps.shape, ref_image_latents.shape, clip_image_embeds.shape)
            self.reference_unet(
                ref_image_latents,
                ref_timesteps,
                encoder_hidden_states=clip_image_embeds,
                return_dict=False,
            )
            self.reference_control_reader.update(self.reference_control_writer)

        model_pred = self.denoising_unet(
            noisy_latents,
            timesteps,
            guidance_fea=pose_fea,
            encoder_hidden_states=clip_image_embeds,
        ).sample

        return model_pred

class VFIMultiRefModel_v8(nn.Module):
    def __init__(
        self,
        reference_unet: UNet2DConditionModel,
        denoising_unet: UNet3DConditionModel,
        seed_guider: PoseGuider,
        pose_guider: PoseGuider,
        reference_control_writer,
        reference_control_reader,
    ):
        super().__init__()
        self.reference_unet = reference_unet
        self.denoising_unet = denoising_unet
        self.pose_guider = pose_guider
        self.seed_guider = seed_guider
        self.reference_control_writer = reference_control_writer
        self.reference_control_reader = reference_control_reader

    def forward(
        self,
        noisy_latents,
        timesteps,
        ref_image_latents,
        clip_image_embeds,
        seed_img = None,
        pose_img = None,
        uncond_fwd: bool = False,
    ):
        if seed_img is not None:
            seed_cond_tensor = seed_img.to(device="cuda")
            seed_fea = self.seed_guider(seed_cond_tensor)
        else:
            seed_fea = None

        if pose_img is not None:
            pose_cond_tensor = pose_img.to(device="cuda")
            pose_fea = self.pose_guider(pose_cond_tensor)
            if seed_fea is None:
                pose_fea = pose_fea
            else:
                pose_fea = seed_fea + pose_fea
        else:
            pose_fea = seed_fea

        # random pick two
        idx_1 = 2
        idx_2 = 11
        ref_image_latents_1 = ref_image_latents[:, :, idx_1]
        ref_image_latents_2 = ref_image_latents[:, :, idx_2]

        length = noisy_latents.shape[2]
        confidence_zero_to_one = torch.linspace(0, 1, steps=length).to(device="cuda")
        confidence_one_to_zero = torch.linspace(1, 0, steps=length).to(device="cuda")
        confidence_1 = torch.cat([confidence_zero_to_one[-idx_1-1:],confidence_one_to_zero[1:-idx_1]])
        confidence_2 = torch.cat([confidence_zero_to_one[-idx_2-1:],confidence_one_to_zero[1:-idx_2]])

        if True: #not uncond_fwd:
            ref_timesteps = torch.zeros_like(timesteps)
            self.reference_unet(
                ref_image_latents_1,
                ref_timesteps,
                encoder_hidden_states=clip_image_embeds,
                return_dict=False,
            )
            self.reference_control_reader.update(self.reference_control_writer, confidence_1)

            self.reference_unet(
                ref_image_latents_2,
                ref_timesteps,
                encoder_hidden_states=clip_image_embeds,
                return_dict=False,
            )
            self.reference_control_reader.update(self.reference_control_writer, confidence_2)

        model_pred = self.denoising_unet(
            noisy_latents,
            timesteps,
            guidance_fea=pose_fea,
            encoder_hidden_states=clip_image_embeds,
        ).sample

        return model_pred


class VFIMultiRefModel_16frames(nn.Module):
    def __init__(
        self,
        reference_unet: UNet2DConditionModel,
        denoising_unet: UNet3DConditionModel,
        seed_guider: PoseGuider,
        pose_guider: PoseGuider,
        reference_control_writer,
        reference_control_reader,
    ):
        super().__init__()
        self.reference_unet = reference_unet
        self.denoising_unet = denoising_unet
        self.pose_guider = pose_guider
        self.seed_guider = seed_guider
        self.reference_control_writer = reference_control_writer
        self.reference_control_reader = reference_control_reader

    def forward(
        self,
        noisy_latents,
        timesteps,
        ref_image_latents,
        clip_image_embeds,
        seed_img = None,
        pose_img = None,
        uncond_fwd: bool = False,
    ):
        if seed_img is not None:
            seed_cond_tensor = seed_img.to(device="cuda")
            seed_fea = self.seed_guider(seed_cond_tensor)
        else:
            seed_fea = None

        if pose_img is not None:
            pose_cond_tensor = pose_img.to(device="cuda")
            pose_fea = self.pose_guider(pose_cond_tensor)
            if seed_fea is None:
                pose_fea = pose_fea
            else:
                pose_fea = seed_fea + pose_fea
        else:
            pose_fea = seed_fea

        # random pick two
        idx_1 = 1
        idx_2 = -1
        ref_image_latents_1 = ref_image_latents[:, :, idx_1]
        ref_image_latents_2 = ref_image_latents[:, :, idx_2]

        length = noisy_latents.shape[2]
        confidence_zero_to_one = torch.linspace(0, 1, steps=length).to(device="cuda")
        confidence_one_to_zero = torch.linspace(1, 0, steps=length).to(device="cuda")
        confidence_1 = confidence_one_to_zero
        confidence_2 = confidence_zero_to_one

        if True: #not uncond_fwd:
            ref_timesteps = torch.zeros_like(timesteps)
            self.reference_unet(
                ref_image_latents_1,
                ref_timesteps,
                encoder_hidden_states=clip_image_embeds,
                return_dict=False,
            )
            self.reference_control_reader.update(self.reference_control_writer, confidence_1)

            self.reference_unet(
                ref_image_latents_2,
                ref_timesteps,
                encoder_hidden_states=clip_image_embeds,
                return_dict=False,
            )
            self.reference_control_reader.update(self.reference_control_writer, confidence_2)

        model_pred = self.denoising_unet(
            noisy_latents,
            timesteps,
            guidance_fea=pose_fea,
            encoder_hidden_states=clip_image_embeds,
        ).sample

        return model_pred



class VFIMultiRefModel_8Time(nn.Module):
    def __init__(
        self,
        reference_unet: UNet2DConditionModel,
        denoising_unet: UNet3DConditionModel,
        seed_guider: PoseGuider,
        pose_guider: PoseGuider,
        reference_control_writer,
        reference_control_reader,
    ):
        super().__init__()
        self.reference_unet = reference_unet
        self.denoising_unet = denoising_unet
        self.pose_guider = pose_guider
        self.seed_guider = seed_guider
        self.reference_control_writer = reference_control_writer
        self.reference_control_reader = reference_control_reader

    def forward(
        self,
        noisy_latents,
        timesteps,
        ref_image_latents,
        clip_image_embeds,
        seed_img = None,
        pose_img = None,
        uncond_fwd: bool = False,
    ):
        if seed_img is not None:
            seed_cond_tensor = seed_img.to(device="cuda")
            seed_fea = self.seed_guider(seed_cond_tensor)
        else:
            seed_fea = None

        if pose_img is not None:
            pose_cond_tensor = pose_img.to(device="cuda")
            pose_fea = self.pose_guider(pose_cond_tensor)
            if seed_fea is None:
                pose_fea = pose_fea
            else:
                pose_fea = seed_fea + pose_fea
        else:
            pose_fea = seed_fea


        length = ref_image_latents.shape[2]
        confidence_zero_to_one = torch.linspace(0, 1, steps=length).to(
            dtype=ref_image_latents.dtype, device=ref_image_latents.device
        )
        confidence_one_to_zero = torch.linspace(1, 0, steps=length).to(
            dtype=ref_image_latents.dtype, device=ref_image_latents.device
        )
        confidence_lst = []
        confidence_lst.append(confidence_one_to_zero.to(
            dtype=ref_image_latents.dtype, device=ref_image_latents.device
        ))
        for idx_1 in range(1,length-1):
            confidence_lst.append(torch.cat([confidence_zero_to_one[-idx_1-1:],confidence_one_to_zero[1:-idx_1]]).to(
            dtype=ref_image_latents.dtype, device=ref_image_latents.device
        ))
        confidence_lst.append(confidence_one_to_zero.to(
            dtype=ref_image_latents.dtype, device=ref_image_latents.device
        ))
        # confidence_lst = [
        #     x.view(1,length,1,1) for x in confidence_lst
        # ]

        if True: #not uncond_fwd:
            ref_timesteps = torch.zeros_like(timesteps)
            for ref_id in [0, 1, 2, 3, 12, 13, 14, 15]:
                self.reference_unet(
                    ref_image_latents[:, :, ref_id],
                    ref_timesteps,
                    encoder_hidden_states=clip_image_embeds,
                    return_dict=False,
                )
                self.reference_control_reader.update(self.reference_control_writer, confidence_lst[ref_id])

        model_pred = self.denoising_unet(
            noisy_latents,
            timesteps,
            guidance_fea=pose_fea,
            encoder_hidden_states=clip_image_embeds,
        ).sample

        return model_pred

class VFIMultiRefModel_1Time(nn.Module):
    def __init__(
        self,
        reference_unet: UNet2DConditionModel,
        denoising_unet: UNet3DConditionModel,
        seed_guider: PoseGuider,
        pose_guider: PoseGuider,
        reference_control_writer,
        reference_control_reader,
    ):
        super().__init__()
        self.reference_unet = reference_unet
        self.denoising_unet = denoising_unet
        self.pose_guider = pose_guider
        self.seed_guider = seed_guider
        self.reference_control_writer = reference_control_writer
        self.reference_control_reader = reference_control_reader

    def forward(
        self,
        noisy_latents,
        timesteps,
        ref_image_latents,
        clip_image_embeds,
        seed_img = None,
        pose_img = None,
        uncond_fwd: bool = False,
    ):
        if seed_img is not None:
            seed_cond_tensor = seed_img.to(device="cuda")
            seed_fea = self.seed_guider(seed_cond_tensor)
        else:
            seed_fea = None

        if pose_img is not None:
            pose_cond_tensor = pose_img.to(device="cuda")
            pose_fea = self.pose_guider(pose_cond_tensor)
            if seed_fea is None:
                pose_fea = pose_fea
            else:
                pose_fea = seed_fea + pose_fea
        else:
            pose_fea = seed_fea


        length = ref_image_latents.shape[2]
        confidence_zero_to_one = torch.linspace(0, 1, steps=length).to(
            dtype=ref_image_latents.dtype, device=ref_image_latents.device
        )
        confidence_one_to_zero = torch.linspace(1, 0, steps=length).to(
            dtype=ref_image_latents.dtype, device=ref_image_latents.device
        )
        confidence_lst = []
        confidence_lst.append(confidence_one_to_zero.to(
            dtype=ref_image_latents.dtype, device=ref_image_latents.device
        ))
        for idx_1 in range(1,length-1):
            confidence_lst.append(torch.cat([confidence_zero_to_one[-idx_1-1:],confidence_one_to_zero[1:-idx_1]]).to(
            dtype=ref_image_latents.dtype, device=ref_image_latents.device
        ))
        confidence_lst.append(confidence_one_to_zero.to(
            dtype=ref_image_latents.dtype, device=ref_image_latents.device
        ))
        # confidence_lst = [
        #     x.view(1,length,1,1) for x in confidence_lst
        # ]

        if True: #not uncond_fwd:
            ref_timesteps = torch.zeros_like(timesteps)
            for ref_id in [0, 1, 2, 3, 12, 13, 14, 15]:
                self.reference_unet(
                    ref_image_latents[:, :, ref_id],
                    ref_timesteps,
                    encoder_hidden_states=clip_image_embeds,
                    return_dict=False,
                )
                self.reference_control_reader.update(self.reference_control_writer, confidence_lst[ref_id])

        model_pred = self.denoising_unet(
            noisy_latents,
            timesteps,
            guidance_fea=pose_fea,
            encoder_hidden_states=clip_image_embeds,
        ).sample

        return model_pred
