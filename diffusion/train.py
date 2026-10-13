import argparse
import logging
import math
import os
import random
import warnings
from pathlib import Path
import json

import wandb
from datetime import datetime
import diffusers
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs
from diffusers import AutoencoderKL, DDIMScheduler
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
from diffusers.utils.import_utils import is_xformers_available
from omegaconf import OmegaConf
from PIL import Image, ImageOps
from tqdm.auto import tqdm
from transformers import CLIPVisionModelWithProjection
from einops import rearrange
from decord import VideoReader

from datasets.vfi_dataset_refinev2 import VideoDataset
from pipelines.pipeline_vfi2video_share_vae import Pose2VideoPipeline
from models.model_vfi import VFIShareVAEModel
from models.pose_guider import PoseGuider
from models.conv_in import ConvIn

from models.unet_2d_condition import UNet2DConditionModel
from models.unet_3d import UNet3DConditionModel
from models.mutual_self_attention import ReferenceAttentionControl

from utils.util import seed_everything, delete_additional_ckpt, compute_snr
from utils.video_utils import save_videos_grid, save_videos_from_pil, concat_pil

warnings.filterwarnings("ignore")
check_min_version("0.10.0.dev0")
logger = get_logger(__name__, log_level="INFO")


def padding_pil(img_pil, img_size):
    # resize a PIL image and zero padding the short edge
    W, H = img_pil.size
    resize_ratio = img_size / max(W, H)
    new_W, new_H = int(W * resize_ratio), int(H * resize_ratio)
    img_pil = img_pil.resize((new_W, new_H))

    left = (img_size - new_W) // 2
    right = img_size - new_W - left
    top = (img_size - new_H) // 2
    bottom = img_size - new_H - top

    padding_border = (left, top, right, bottom)
    img_pil = ImageOps.expand(img_pil, border=padding_border, fill=0)

    return img_pil


def log_validation(
        cfg,
        vae,
        image_enc,
        model,
        scheduler,
        accelerator,
        width,
        height,
        seed=42,
        dtype=torch.float32,
        global_step=1,
    ):
    logger.info("Running validation ...")
    unwrap_model = accelerator.unwrap_model(model).eval()
    denoising_unet = unwrap_model.denoising_unet
    reference_unet = unwrap_model.reference_unet
    pose_guider = unwrap_model.pose_guider
    seed_guider = unwrap_model.seed_guider
    generator = torch.manual_seed(seed)
    vae = vae.to(dtype=dtype)
    image_enc = image_enc.to(dtype=dtype)

    pipeline = Pose2VideoPipeline(
        vae=vae,
        image_encoder=image_enc,
        reference_unet=reference_unet,
        denoising_unet=denoising_unet,
        pose_guider=pose_guider,
        seed_guider=seed_guider,
        scheduler=scheduler,
    )
    pipeline = pipeline.to(accelerator.device)

    vid_meta = []
    for data_meta_path in cfg.data.meta_paths:
        vid_meta.extend(json.load(open(data_meta_path, "r")))
    all_test_cases = [item for item in vid_meta if item.get("mode") == "test"]

    current_device_id = accelerator.process_index
    pil_images = []

    if cfg.wandb_project == "debug":
        all_test_cases = all_test_cases[:2] + all_test_cases[-2:]
    else:
        all_test_cases = all_test_cases[:8] + all_test_cases[-8:]

    with torch.no_grad():
        for idx, test_case in enumerate(all_test_cases):
            # if idx > 1: break
            if idx % accelerator.num_processes == current_device_id:
                video_path = test_case["video_path"]
                kps_pth = test_case["kps_path"].replace(".mp4", "_weighted_heatmap.mp4")
                rgb_video = VideoReader(video_path)
                kps_video = VideoReader(kps_pth)

                if idx >= len(all_test_cases) // 2:
                    # for dance video we pick from 8s
                    ref_id = 120
                else:
                    ref_id = 15

                start_g = ref_id - cfg.data.seed_frames
                end_s = ref_id + cfg.data.generated_frames
                end_g = ref_id + cfg.data.generated_frames + cfg.data.seed_frames
                # print(start_g, ref_id, end_s, end_g)

                ref_img = rgb_video[ref_id-1]
                ref_img_pil = Image.fromarray(ref_img.asnumpy())
                w, h = ref_img_pil.size

                tgt_vidpil_lst = []
                tgt_guid_vid_list = []
                pose_guid_vid_list = []
                for index in range(start_g, ref_id):
                    img = rgb_video[index]
                    pil_img = Image.fromarray(img.asnumpy())
                    tgt_guid_pil = pil_img.copy()
                    pose_guid_pil = Image.new(pil_img.mode, pil_img.size, 0)
                    pose_guid_vid_list.append(pose_guid_pil)
                    tgt_vidpil_lst.append(pil_img)
                    tgt_guid_vid_list.append(tgt_guid_pil)

                for index in range(ref_id, end_s):
                    if index >= ref_id + cfg.data.generated_frames // 2 :
                        img_dup = rgb_video[end_s]
                        pil_img_dup = Image.fromarray(img_dup.asnumpy())
                        tgt_guid_vid_list.append(pil_img_dup)
                    else:
                        img_dup = rgb_video[ref_id-1]
                        pil_img_dup = Image.fromarray(img_dup.asnumpy())
                        tgt_guid_vid_list.append(pil_img_dup)
                    pose_guid_pil = Image.fromarray(kps_video[index].asnumpy())
                    pose_guid_vid_list.append(pose_guid_pil)

                    img = rgb_video[index]
                    pil_img = Image.fromarray(img.asnumpy())
                    tgt_vidpil_lst.append(pil_img)

                for index in range(end_s, end_g):
                    img = rgb_video[index]
                    pil_img = Image.fromarray(img.asnumpy())
                    tgt_guid_pil = pil_img.copy()
                    pose_guid_pil = Image.new(pil_img.mode, pil_img.size, 0)
                    pose_guid_vid_list.append(pose_guid_pil)
                    tgt_vidpil_lst.append(pil_img)
                    tgt_guid_vid_list.append(tgt_guid_pil)

                # avoid out of memory
                if global_step == 1:
                    factor_test = 1
                else:
                    factor_test = int((global_step/1000)%3 + 1)
                print(factor_test, 256*factor_test)
                ref_img_pil = padding_pil(ref_img_pil, 256*factor_test)
                tgt_guid_vid_list = [padding_pil(x, 256*factor_test) for x in tgt_guid_vid_list]
                tgt_vidpil_lst = [padding_pil(x, 256*factor_test) for x in tgt_vidpil_lst]
                pose_guid_vid_list = [padding_pil(x, 256*factor_test) for x in pose_guid_vid_list]
                w = h = 256*factor_test

                generator = torch.manual_seed(seed)
                generator = torch.manual_seed(seed)
                video_tensor = pipeline(
                    ref_img_pil,
                    pose_guid_vid_list,
                    tgt_guid_vid_list,
                    w,
                    h,
                    cfg.data.sample_frames,
                    cfg.validation.denoising_steps,
                    cfg.validation.guidance_scale,
                    generator=generator,
                ).videos
                video_tensor = video_tensor[0, ...].permute(1, 2, 3, 0).cpu().numpy()

                video_pil_lst = []
                for frame_idx, image_tensor in enumerate(video_tensor):
                    result_img_pil = Image.fromarray((image_tensor * 255).astype(np.uint8))
                    if frame_idx >= cfg.data.seed_frames and frame_idx < cfg.data.seed_frames+cfg.data.generated_frames:
                        result_pil_lst = [pose_guid_vid_list[frame_idx], tgt_guid_vid_list[frame_idx], result_img_pil, tgt_vidpil_lst[frame_idx], result_img_pil]
                    else:
                        result_pil_lst = [pose_guid_vid_list[frame_idx], tgt_guid_vid_list[frame_idx], result_img_pil, tgt_vidpil_lst[frame_idx], tgt_vidpil_lst[frame_idx]]
                    concated_pil = concat_pil(result_pil_lst)
                    video_pil_lst.append(concated_pil)
                pil_images.append({"name": f"{idx}", "video": video_pil_lst})

    unwrap_model.train()
    vae = vae.to(dtype=torch.float16)
    image_enc = image_enc.to(dtype=torch.float16)


    del pipeline
    unwrap_model.reference_control_writer = ReferenceAttentionControl(
        reference_unet, do_classifier_free_guidance=False, mode="write", fusion_blocks="full"
    )
    unwrap_model.reference_control_reader = ReferenceAttentionControl(
        denoising_unet, do_classifier_free_guidance=False, mode="read", fusion_blocks="full"
    )
    torch.cuda.empty_cache()
    return pil_images


def train_val_function(
    accelerator, weight_dtype, vae, train_noise_scheduler, image_enc,
    trainable_params, optimizer, lr_scheduler, train_loss, model, batch, cfg, mode="train"
):
    if mode == "train":
        model.train()
    else:
        model.eval()
    with accelerator.accumulate(model):
        # Convert videos to latent space
        pixel_values_vid = batch["tgt_vid"].to(weight_dtype)
        tgt_guid_videos = batch["tgt_guid_vid"].to(weight_dtype)
        heatmap_vid = batch["heatmap_vid"].to(weight_dtype)

        with torch.no_grad():
            video_length = pixel_values_vid.shape[1]
            pixel_values_vid = rearrange(
                pixel_values_vid, "b f c h w -> (b f) c h w"
            )
            latents = vae.encode(pixel_values_vid).latent_dist.sample()
            latents = rearrange(
                latents, "(b f) c h w -> b c f h w", f=video_length
            )
            latents = latents * 0.18215

            tgt_guid_videos = rearrange(
                tgt_guid_videos, "b f c h w -> (b f) c h w"
            )
            latents_seeds = vae.encode(tgt_guid_videos).latent_dist.sample()
            latents_seeds = rearrange(
                latents_seeds, "(b f) c h w -> b c f h w", f=video_length
            )
            latents_seeds = latents_seeds * 0.18215


        bs, tt, cc, hh, ww = heatmap_vid.shape
        heatmap_vid = heatmap_vid.view(-1, cc, hh, ww)
        heatmap_vid = F.interpolate(heatmap_vid, scale_factor=1/8, mode='bilinear', align_corners=False)
        heatmap_vid = heatmap_vid.view(bs, tt, cc, hh//8, ww//8)
        heatmap_vid = heatmap_vid.mean(dim=2).unsqueeze(1)


        noise = torch.randn_like(latents)
        if cfg.noise_offset > 0:
            noise += cfg.noise_offset * torch.randn(
                (latents.shape[0], latents.shape[1], 1, 1, 1),
                device=latents.device,
            )
        bsz = latents.shape[0]
        # Sample a random timestep for each video
        timesteps = torch.randint(
            0,
            train_noise_scheduler.num_train_timesteps,
            (bsz,),
            device=latents.device,
        )
        timesteps = timesteps.long()

        uncond_fwd = random.random() < cfg.uncond_ratio

        if uncond_fwd:
            pose_guid_videos = torch.zeros_like(batch["pose_guid_vid"]).to(device=latents_seeds.device, dtype=latents_seeds.dtype)
        else:
            pose_guid_videos = batch["pose_guid_vid"]  # (bs, f, c, H, W)
        pose_guid_videos = pose_guid_videos.transpose(
            1, 2
        )  # (bs, c, f, H, W)

        clip_image_list = []
        ref_image_list = []
        for batch_idx, (ref_img, clip_img) in enumerate(
            zip(
                batch["ref_img"],
                batch["clip_img"],
            )
        ):
            if uncond_fwd:
                clip_image_list.append(torch.zeros_like(clip_img))
            else:
                clip_image_list.append(clip_img)
            ref_image_list.append(ref_img)

        with torch.no_grad():
            ref_img = torch.stack(ref_image_list, dim=0).to(
                dtype=vae.dtype, device=vae.device
            )
            ref_image_latents = vae.encode(
                ref_img
            ).latent_dist.sample()  # (bs, d, 64, 64)
            ref_image_latents = ref_image_latents * 0.18215

            clip_img = torch.stack(clip_image_list, dim=0).to(
                dtype=image_enc.dtype, device=image_enc.device
            )
            clip_img = clip_img.to(device="cuda", dtype=weight_dtype)
            clip_image_embeds = image_enc(
                clip_img.to("cuda", dtype=weight_dtype)
            ).image_embeds
            clip_image_embeds = clip_image_embeds.unsqueeze(1)  # (bs, 1, d)

        # print("1", clip_image_embeds.shape, ref_image_latents.shape, latents.shape, tgt_guid_videos.shape)
        # add noise
        noisy_latents = train_noise_scheduler.add_noise(
            latents, noise, timesteps
        )

        # Get the target for loss depending on the prediction type
        if train_noise_scheduler.prediction_type == "epsilon":
            target = noise
        elif train_noise_scheduler.prediction_type == "v_prediction":
            target = train_noise_scheduler.get_velocity(
                latents, noise, timesteps
            )
        else:
            raise ValueError(
                f"Unknown prediction type {train_noise_scheduler.prediction_type}"
            )

        model_pred = model(
            noisy_latents,
            timesteps,
            ref_image_latents,
            clip_image_embeds,
            seed_img = latents_seeds,
            pose_img = pose_guid_videos,
            uncond_fwd=uncond_fwd,
        )

        if cfg.snr_gamma == 0:
            loss = F.mse_loss(
                model_pred.float(), target.float(), reduction="mean"
            )
        else:
            snr = compute_snr(train_noise_scheduler, timesteps)
            if train_noise_scheduler.config.prediction_type == "v_prediction":
                # Velocity objective requires that we add one to SNR values before we divide by them.
                snr = snr + 1
            mse_loss_weights = (
                torch.stack(
                    [snr, cfg.snr_gamma * torch.ones_like(timesteps)], dim=1
                ).min(dim=1)[0]
                / snr
            )
            loss = F.mse_loss(
                model_pred.float()*(1.0+heatmap_vid.float()), target.float()*(1.0+heatmap_vid.float()), reduction="none"
            )
            loss = (
                loss.mean(dim=list(range(1, len(loss.shape))))
                * mse_loss_weights
            )
            loss = loss.mean()

        # Gather the losses across all processes for logging (if we use distributed training).
        avg_loss = accelerator.gather(loss.repeat(cfg.data.train_bs)).mean()
        train_loss += avg_loss.item() / cfg.solver.gradient_accumulation_steps

        if mode == "train":
            # Backpropagate
            accelerator.backward(loss)
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(
                    trainable_params,
                    cfg.solver.max_grad_norm,
                )
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
    unwrap_model = accelerator.unwrap_model(model)
    unwrap_model.reference_control_reader.clear()
    unwrap_model.reference_control_writer.clear()
    return train_loss, loss

def main(cfg):
    os.environ.setdefault("WANDB_MODE", "disabled")
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.solver.gradient_accumulation_steps,
        mixed_precision=cfg.solver.mixed_precision,
        log_with="wandb",
        project_dir=cfg.wandb_log_dir,
        kwargs_handlers=[kwargs],
    )

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=True)

    if accelerator.is_main_process:
        save_dir = os.path.join(cfg.output_dir, cfg.exp_name)
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(os.path.join(save_dir, 'checkpoints'), exist_ok=True)
        os.makedirs(os.path.join(save_dir, 'sanity_check'), exist_ok=True)
        os.makedirs(os.path.join(save_dir, 'saved_models'), exist_ok=True)
        os.makedirs(os.path.join(save_dir, 'validation'), exist_ok=True)

    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    if cfg.seed is not None:
        seed_everything(cfg.seed)

    if cfg.weight_dtype == "fp16":
        weight_dtype = torch.float16
    elif cfg.weight_dtype == "fp32":
        weight_dtype = torch.float32
    else:
        raise ValueError(
            f"Do not support weight dtype: {cfg.weight_dtype} during training"
        )

    sched_kwargs = OmegaConf.to_container(cfg.noise_scheduler_kwargs)
    if cfg.enable_zero_snr:
        sched_kwargs.update(
            rescale_betas_zero_snr=True,
            timestep_spacing="trailing",
            prediction_type="v_prediction",
        )
    val_noise_scheduler = DDIMScheduler(**sched_kwargs)
    sched_kwargs.update({"beta_schedule": "scaled_linear"})
    train_noise_scheduler = DDIMScheduler(**sched_kwargs)

    vae = AutoencoderKL.from_pretrained(cfg.vae_model_path).to(
        "cuda", dtype=weight_dtype
    )

    reference_unet = UNet2DConditionModel.from_pretrained(
        cfg.base_model_path,
        subfolder="unet",
    ).to(device="cuda")

    denoising_unet = UNet3DConditionModel.from_pretrained_2d(
        cfg.base_model_path,
        cfg.mm_path,
        subfolder="unet",
        unet_additional_kwargs=OmegaConf.to_container(
            cfg.unet_additional_kwargs
        ),
    ).to(device="cuda")

    image_enc = CLIPVisionModelWithProjection.from_pretrained(
        cfg.image_encoder_path,
    ).to(dtype=weight_dtype, device="cuda")

    if cfg.pose_guider_pretrain:
        pose_guider = PoseGuider(
            conditioning_embedding_channels=320, block_out_channels=(16, 32, 96, 256),
            conditioning_channels=3,
        ).to(device="cuda")
        # load pretrained controlnet-openpose params for pose_guider
        controlnet_openpose_state_dict = torch.load(cfg.controlnet_openpose_path)
        state_dict_to_load = {}
        for k in controlnet_openpose_state_dict.keys():
            if k.startswith("controlnet_cond_embedding.") and k.find("conv_out") < 0:
                new_k = k.replace("controlnet_cond_embedding.", "")
                # skip first conv layer
                # if new_k != "conv_in.weight":
                state_dict_to_load[new_k] = controlnet_openpose_state_dict[k]
        miss, _ = pose_guider.load_state_dict(state_dict_to_load, strict=False)
        logger.info(f"Missing key for pose guider: {len(miss)}")
    else:
        pose_guider = PoseGuider(
            conditioning_embedding_channels=320, block_out_channels=(16, 32, 96, 256),
            conditioning_channels=3,
        ).to(device="cuda")

    seed_guider = ConvIn(conditioning_embedding_channels=320, conditioning_channels=4)

    # Freeze
    vae.requires_grad_(False)
    image_enc.requires_grad_(False)
    denoising_unet.requires_grad_(True)
    pose_guider.requires_grad_(True)
    for name, param in reference_unet.named_parameters():
        if "up_blocks.3" in name:
            param.requires_grad_(False)
        else:
            param.requires_grad_(True)
    for name, param in denoising_unet.named_parameters():
        if "conv_in" in name:
            # print(name)
            param.requires_grad_(False)
    seed_guider.requires_grad_(True)

    reference_control_writer = ReferenceAttentionControl(
        reference_unet,
        do_classifier_free_guidance=False,
        mode="write",
        fusion_blocks="full",
    )
    reference_control_reader = ReferenceAttentionControl(
        denoising_unet,
        do_classifier_free_guidance=False,
        mode="read",
        fusion_blocks="full",
    )

    model = VFIShareVAEModel(
        reference_unet,
        denoising_unet,
        seed_guider,
        pose_guider,
        reference_control_writer,
        reference_control_reader,
    )

    if cfg.stage1_ckpt_dir:
        old_ckpt = torch.load(
            os.path.join(cfg.stage1_ckpt_dir, f"checkpoint-{cfg.stage1_ckpt_step}", "pytorch_model.bin"),
            map_location="cpu",
        )
        model.load_state_dict(old_ckpt)
        del old_ckpt
        logger.info(f"Loaded stage1 models from {cfg.stage1_ckpt_dir}, step={cfg.stage1_ckpt_step}")


    if cfg.solver.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            reference_unet.enable_xformers_memory_efficient_attention()
            denoising_unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError(
                "xformers is not available. Make sure it is installed correctly"
            )

    if cfg.solver.gradient_checkpointing:
        reference_unet.enable_gradient_checkpointing()
        denoising_unet.enable_gradient_checkpointing()

    if cfg.solver.scale_lr:
        learning_rate = (
            cfg.solver.learning_rate
            * cfg.solver.gradient_accumulation_steps
            * cfg.data.train_bs
            * accelerator.num_processes
        )
    else:
        learning_rate = cfg.solver.learning_rate

    if cfg.solver.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    trainable_params = list(filter(lambda p: p.requires_grad, model.parameters()))
    logger.info(f"Total trainable params {len(trainable_params)}")
    optimizer = optimizer_cls(
        trainable_params,
        lr=learning_rate,
        betas=(cfg.solver.adam_beta1, cfg.solver.adam_beta2),
        weight_decay=cfg.solver.adam_weight_decay,
        eps=cfg.solver.adam_epsilon,
    )
    lr_scheduler = get_scheduler(
        cfg.solver.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=cfg.solver.lr_warmup_steps
        * cfg.solver.gradient_accumulation_steps,
        num_training_steps=cfg.solver.max_train_steps
        * cfg.solver.gradient_accumulation_steps,
    )

    train_dataset = VideoDataset(
        image_size=cfg.data.image_size,
        data_meta_paths=cfg.data.meta_paths,
        sample_frames=cfg.data.sample_frames,
        sample_rate=cfg.data.sample_rate,
        seed_frames=cfg.data.seed_frames,
        generated_frames=cfg.data.generated_frames,
        aug_type=cfg.data.aug_type,
        mode="train",
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=cfg.data.train_bs, shuffle=True, num_workers=16
    )

    val_dataset = VideoDataset(
        image_size=cfg.data.image_size,
        data_meta_paths=cfg.data.meta_paths,
        sample_frames=cfg.data.sample_frames,
        sample_rate=cfg.data.sample_rate,
        seed_frames=cfg.data.seed_frames,
        generated_frames=cfg.data.generated_frames,
        aug_type=cfg.data.aug_type,
        mode="test",
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=cfg.data.train_bs, shuffle=False, num_workers=16
    )

    model, optimizer, train_dataloader, val_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, val_dataloader, lr_scheduler
    )

    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / cfg.solver.gradient_accumulation_steps
    )
    num_train_epochs = math.ceil(
        cfg.solver.max_train_steps / num_update_steps_per_epoch
    )

    if accelerator.is_main_process:
        run_time = datetime.now().strftime("%Y%m%d-%H%M")
        accelerator.init_trackers(
            cfg.wandb_project,
            init_kwargs={"wandb": {"name": cfg.exp_name + "_" + run_time, "dir":cfg.wandb_log_dir}},
        )
        config_dict = OmegaConf.to_container(cfg)
        wandb.config.update(config_dict)

    logger.info("Start training ...")
    logger.info(f"Num Samples: {len(train_dataset)}")
    logger.info(f"Train Batchsize: {cfg.data.train_bs}")
    logger.info(f"Num Epochs: {num_train_epochs}")
    logger.info(f"Total Steps: {cfg.solver.max_train_steps}")

    global_step, first_epoch, resume_step = 0, 0, 0
    if cfg.resume_from_checkpoint:
        accelerator.load_state(cfg.resume_from_checkpoint)
        global_step = int(Path(cfg.resume_from_checkpoint).name.split("-")[-1])
        first_epoch, epoch_step = divmod(global_step, num_update_steps_per_epoch)
        resume_step = epoch_step * cfg.solver.gradient_accumulation_steps
        accelerator.print(f"Resuming from checkpoint {cfg.resume_from_checkpoint}, step={global_step}")

    progress_bar = tqdm(
        range(global_step, cfg.solver.max_train_steps),
        disable=not accelerator.is_local_main_process,
    )
    progress_bar.set_description("Steps")

    for epoch in range(first_epoch, num_train_epochs):
        train_loss = 0.0
        epoch_dataloader = train_dataloader
        if epoch == first_epoch and resume_step:
            epoch_dataloader = accelerator.skip_first_batches(train_dataloader, resume_step)
        for _, batch in enumerate(epoch_dataloader):
            if global_step >= cfg.solver.max_train_steps:
                break
            train_loss, loss = train_val_function(
                accelerator, weight_dtype, vae, train_noise_scheduler, image_enc,
                trainable_params, optimizer, lr_scheduler, train_loss, model, batch, cfg, mode="train"
            )

            if global_step % cfg.validation.val_loss_steps == 0:
                accelerator.wait_for_everyone()
                val_loss = 0.
                for _, val_batch in enumerate(val_dataloader):
                    with torch.no_grad():
                        val_loss, _ = train_val_function(
                                accelerator, weight_dtype, vae, val_noise_scheduler, image_enc,
                                trainable_params, optimizer, lr_scheduler, val_loss, model, val_batch, cfg, mode="val"
                            )
                val_loss_avg = val_loss/len(val_dataloader)
                accelerator.log({"val_loss": val_loss_avg}, step=global_step)

            save_dir = f'{cfg.output_dir}/{cfg.exp_name}'
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0
                #　save checkpoints
                if global_step % cfg.validation.validation_steps == 0 or global_step == cfg.solver.max_train_steps:
                    if accelerator.is_main_process:
                        save_path = os.path.join(save_dir, f"checkpoint-{global_step}")
                        delete_additional_ckpt(save_dir, 1)
                        accelerator.save_state(save_path)

                #  sanity check
                if global_step == 1:
                    ref_forcheck = batch['ref_img'] * 0.5 + 0.5
                    img_forcheck = batch['tgt_vid'] * 0.5 + 0.5
                    guid_forcheck = batch['tgt_guid_vid'] * 0.5 + 0.5
                    pose_forcheck = batch['pose_guid_vid']

                    ref_forcheck = ref_forcheck.unsqueeze(2).repeat(1, 1, cfg.data.sample_frames, 1, 1)
                    img_forcheck = rearrange(img_forcheck, 'b f c h w -> b c f h w')
                    guid_forcheck = rearrange(guid_forcheck, 'b f c h w -> b c f h w')
                    pose_forcheck = rearrange(pose_forcheck, 'b f c h w -> b c f h w')

                    video_forcheck = torch.cat([ref_forcheck, img_forcheck, guid_forcheck, pose_forcheck], dim=0).cpu()
                    save_videos_grid(video_forcheck, f'{cfg.output_dir}/{cfg.exp_name}/sanity_check/data-{global_step:06d}-rank{accelerator.device.index}.gif', fps=30, n_rows=4)

                if global_step % cfg.validation.validation_steps == 0 or global_step == 1:
                    sample_dicts = log_validation(
                        cfg=cfg,
                        vae=vae,
                        image_enc=image_enc,
                        model=model,
                        scheduler=val_noise_scheduler,
                        accelerator=accelerator,
                        width=cfg.data.image_size,
                        height=cfg.data.image_size,
                        seed=cfg.seed,
                        global_step=global_step,
                    )

                    image_save_dir = f"{cfg.wandb_log_dir}/{cfg.exp_name}/{global_step:06d}/"
                    if not os.path.exists(image_save_dir):
                        os.makedirs(image_save_dir, exist_ok=True)

                    for sample_id, sample_dict in enumerate(sample_dicts):
                        sample_name = sample_dict["name"]
                        img = sample_dict["video"]
                        # print(img[0].size[0], img[0].size[1])
                        if img[0].size[0] >= img[0].size[1]:
                            img_path = f"{image_save_dir}{global_step:06d}-{sample_name}-higher.mp4"
                        else:
                            img_path = f"{image_save_dir}{global_step:06d}-{sample_name}-wider.mp4"
                        save_videos_from_pil(img, img_path, fps=6)

                    accelerator.wait_for_everyone()
                    if accelerator.is_main_process:
                        images_to_log_higher = []
                        images_to_log_wider = []
                        for img_path in os.listdir(image_save_dir):
                            if img_path.endswith("-higher.mp4"):
                                wandb_img = wandb.Video(image_save_dir+img_path, caption=f"{global_step:06d}-{img_path}")
                                images_to_log_higher.append(wandb_img)
                            elif img_path.endswith("-wider.mp4"):
                                wandb_img = wandb.Video(image_save_dir+img_path, caption=f"{global_step:06d}-{img_path}")
                                images_to_log_wider.append(wandb_img)


                        wandb.log({
                                    "images_higher": images_to_log_higher,
                                    "images_wider": images_to_log_wider,
                                    "iteration": global_step
                                })

            logs = {
                "step_loss": loss.detach().item(),
                "lr": lr_scheduler.get_last_lr()[0],
                "stage": 2,
            }
            progress_bar.set_postfix(**logs)

            if global_step >= cfg.solver.max_train_steps:
                break
        if global_step >= cfg.solver.max_train_steps:
            break

    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="diffusion/configs/train.yaml")
    args = parser.parse_args()

    if args.config[-5:] == ".yaml":
        config = OmegaConf.load(args.config)
    else:
        raise ValueError("Do not support this format config file")

    main(config)
