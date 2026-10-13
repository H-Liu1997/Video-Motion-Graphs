import argparse
import logging
import os
import warnings
from pathlib import Path
import json

import wandb
from datetime import datetime
import diffusers
import numpy as np
import torch
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs
from diffusers import AutoencoderKL, DDIMScheduler
from diffusers.utils import check_min_version
from diffusers.utils.import_utils import is_xformers_available
from omegaconf import OmegaConf
from PIL import Image
from transformers import CLIPVisionModelWithProjection
from decord import VideoReader

from pipelines.pipeline_vfi2video_share_vae import Pose2VideoPipeline
from models.model_vfi import VFIShareVAEModel
from models.pose_guider import PoseGuider
from models.conv_in import ConvIn

from models.unet_2d_condition import UNet2DConditionModel
from models.unet_3d import UNet3DConditionModel
from models.mutual_self_attention import ReferenceAttentionControl

from utils.util import seed_everything
from utils.video_utils import concat_pil

warnings.filterwarnings("ignore")
check_min_version("0.10.0.dev0")
logger = get_logger(__name__, log_level="INFO")

def save_videos_from_pil(pil_images, path, fps=8, bitrate=None):
    import av

    save_fmt = Path(path).suffix
    os.makedirs(os.path.dirname(path), exist_ok=True)
    width, height = pil_images[0].size

    if save_fmt == ".mp4":
        codec = "libx264"
        container = av.open(path, "w")
        stream = container.add_stream(codec, rate=fps, )

        stream.width = width
        stream.height = height
        if bitrate is not None:
            stream.bit_rate = bitrate

        for pil_image in pil_images:
            # pil_image = Image.fromarray(image_arr).convert("RGB")
            av_frame = av.VideoFrame.from_image(pil_image)
            container.mux(stream.encode(av_frame))
        container.mux(stream.encode())
        container.close()

    elif save_fmt == ".gif":
        pil_images[0].save(
            fp=path,
            format="GIF",
            append_images=pil_images[1:],
            save_all=True,
            duration=(1 / fps * 1000),
            loop=0,
        )
    else:
        raise ValueError("Unsupported file type. Use .mp4 or .gif.")


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

    # if cfg.wandb_project == "debug":
    #     all_test_cases = all_test_cases[:2] + all_test_cases[-2:]
    # else:
    #     all_test_cases = all_test_cases[:8] + all_test_cases[-8:]

    with torch.no_grad():
        for idx, test_case in enumerate(all_test_cases):
            # if idx > 1: break
            if idx % accelerator.num_processes == current_device_id:
                video_path = test_case["video_path"]
                kps_pth = test_case["kps_path"]
                rgb_video = VideoReader(video_path)
                kps_video = VideoReader(kps_pth)

                # if idx >= len(all_test_cases) // 2:
                #     # for dance video we pick from 8s
                #     ref_id = 120
                # else:
                ref_id = 3

                start_g = ref_id - cfg.data.seed_frames
                end_s = ref_id + cfg.data.generated_frames
                end_g = ref_id + cfg.data.generated_frames + cfg.data.seed_frames

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

                ori_w = w
                ori_h = h

                # generator = torch.manual_seed(seed)
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
                video_pil_lst_original_size = []
                gap_w = (w - ori_w) // 2
                gap_h = (h - ori_h) // 2
                for frame_idx, image_tensor in enumerate(video_tensor):
                    result_img_pil = Image.fromarray((image_tensor * 255).astype(np.uint8))
                    result_img_pil_2 = result_img_pil.copy()
                    result_img_np = np.array(result_img_pil_2)
                    result_img_np_ori = result_img_np[gap_h:gap_h+ori_h, gap_w:gap_w+ori_w]
                    result_img_pil_ori = Image.fromarray(result_img_np_ori)
                    # print(result_img_pil_ori.size)
                    if frame_idx >= cfg.data.seed_frames and frame_idx < cfg.data.seed_frames+cfg.data.generated_frames:
                        result_pil_lst = [pose_guid_vid_list[frame_idx], tgt_guid_vid_list[frame_idx], result_img_pil, tgt_vidpil_lst[frame_idx], result_img_pil]
                    else:
                        result_pil_lst = [pose_guid_vid_list[frame_idx], tgt_guid_vid_list[frame_idx], result_img_pil, tgt_vidpil_lst[frame_idx], tgt_vidpil_lst[frame_idx]]
                    concated_pil = concat_pil(result_pil_lst)
                    video_pil_lst_original_size.append(result_img_pil_ori)
                    video_pil_lst.append(concated_pil)
                pil_images.append({"name": f"{test_case['video_path'][-14:-4]}", "video": video_pil_lst})
                # clip = ImageSequenceClip(video_pil_lst_original_size, fps=24)
                # clip.write_videofile(video_path.replace("_rgb_", "_blend_res_"), codec='libx264', bitrate="4000k")
                output_path = video_path.replace("_rgb_", "_blend_res_")
                if output_path == video_path:
                    output_path = str(Path(cfg.output_dir) / cfg.exp_name / f"{Path(video_path).stem}_blend_res.mp4")
                save_videos_from_pil(video_pil_lst_original_size, output_path, bitrate=4000000)

    unwrap_model.train()
    vae = vae.to(dtype=torch.float16)
    image_enc = image_enc.to(dtype=torch.float16)

    del pipeline
    torch.cuda.empty_cache()
    return pil_images

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

    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    if cfg.seed is not None:
        seed_everything(cfg.seed)

    sched_kwargs = OmegaConf.to_container(cfg.noise_scheduler_kwargs)
    if cfg.enable_zero_snr:
        sched_kwargs.update(
            rescale_betas_zero_snr=True,
            timestep_spacing="trailing",
            prediction_type="v_prediction",
        )
    val_noise_scheduler = DDIMScheduler(**sched_kwargs)
    sched_kwargs.update({"beta_schedule": "scaled_linear"})

    vae = AutoencoderKL.from_pretrained(cfg.vae_model_path).to(
        "cuda", dtype=torch.float32,
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
    ).to(dtype=torch.float32, device="cuda")
    pose_guider = PoseGuider(
        conditioning_embedding_channels=320, block_out_channels=(16, 32, 96, 256),
        conditioning_channels=3,
    ).to(device="cuda")
    seed_guider = ConvIn(conditioning_embedding_channels=320, conditioning_channels=4)

    vae.requires_grad_(False)
    image_enc.requires_grad_(False)
    denoising_unet.requires_grad_(False)
    pose_guider.requires_grad_(False)
    seed_guider.requires_grad_(False)

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

    old_ckpt = torch.load(
        os.path.join(cfg.stage1_ckpt_dir, f"checkpoint-{cfg.stage1_ckpt_step}", "pytorch_model.bin"),
        map_location="cpu",
    )
    model.load_state_dict(old_ckpt)
    del old_ckpt
    logger.info(f"Loaded stage1 models from {cfg.stage1_ckpt_dir}, step={cfg.stage1_ckpt_step}")

    model = accelerator.prepare(
        model
    )

    if cfg.solver.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            reference_unet.enable_xformers_memory_efficient_attention()
            denoising_unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError(
                "xformers is not available. Make sure it is installed correctly"
            )

    if accelerator.is_main_process:
        run_time = datetime.now().strftime("%Y%m%d-%H%M")
        accelerator.init_trackers(
            cfg.wandb_project,
            init_kwargs={"wandb": {"name": cfg.exp_name + "_" + run_time, "dir":cfg.wandb_log_dir}},
        )
        config_dict = OmegaConf.to_container(cfg)
        wandb.config.update(config_dict)

    accelerator.wait_for_everyone()
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
    )
    global_step = 0
    image_save_dir = f"{cfg.wandb_log_dir}/{cfg.exp_name}/{global_step:06d}/"
    if not os.path.exists(image_save_dir):
        os.makedirs(image_save_dir, exist_ok=True)

    for sample_id, sample_dict in enumerate(sample_dicts):
        sample_name = sample_dict["name"]
        img = sample_dict["video"]
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
    accelerator.wait_for_everyone()
    accelerator.end_training()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="diffusion/configs/inference.yaml")
    parser.add_argument("--json", type=str, default="")
    args = parser.parse_args()

    if args.config[-5:] == ".yaml":
        config = OmegaConf.load(args.config)
    else:
        raise ValueError("Do not support this format config file")

    if args.json:
        config.data.meta_paths = [args.json]

    main(config)
