import gc
import json
import os
from pathlib import Path

import cv2
import torch
from loguru import logger
from torchvision import transforms
from torchvision.models.segmentation import deeplabv3_resnet101
from tqdm import tqdm

from .tools.dwpose import DWposeDetector
from .tools.extract_dwpose_from_vid import process_single_video, process_video


def segment_video(video_path, output_path, weights_path):
    model = deeplabv3_resnet101(weights=None, weights_backbone=None, aux_loss=True)
    model.load_state_dict(torch.load(weights_path, map_location="cpu"), strict=True)
    model.eval().to("cuda")
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    cap = cv2.VideoCapture(str(video_path))
    fps = round(cap.get(cv2.CAP_PROP_FPS))
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size, False)
    with torch.inference_mode(), tqdm(total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), desc="person segmentation") as progress:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            output = model(preprocess(rgb).unsqueeze(0).to("cuda"))["out"][0]
            inverse_mask = (output.argmax(dim=0) != 15).byte().cpu().numpy() * 255
            writer.write(inverse_mask)
            progress.update(1)
    cap.release()
    writer.release()
    del model
    torch.cuda.empty_cache()


def prepare_video(args):
    video = Path(args.data_path) / (args.video_ori_name + ".mp4")
    output = args.preprocessed_path
    output.mkdir(parents=True, exist_ok=True)
    prefix = output / args.video_ori_name
    pose_video = prefix.with_suffix(".mp4")
    if not pose_video.exists():
        logger.info("Extracting DWPose from {}", video)
        detector = DWposeDetector().to("cuda", model_dir=Path(args.pretrained_weights) / "DWPose")
        logger.info("DWPose execution providers: {}", detector.pose_estimation.session_pose.get_providers())
        process_single_video(args.video_ori_name, detector, args.data_path, str(output) + os.sep)
        del detector
        gc.collect()
    weighted = Path(str(prefix) + "_weighted_heatmap.mp4")
    mask = Path(str(prefix) + "_hands_face.mp4")
    if not weighted.exists() or not mask.exists():
        process_video(str(output) + os.sep, args.video_ori_name)
    segmentation = Path(str(prefix) + "_segmask_inverse.mp4")
    if not segmentation.exists():
        segment_video(video, segmentation, Path(args.pretrained_weights) / "segmentation/deeplabv3_resnet101_coco-586e9e4e.pth")
    metadata = [
        {"video_path": str(video), "kps_path": str(pose_video), "mask_path": str(mask), "mode": mode}
        for mode in ("train", "test")
    ]
    cache = Path(args.cache_path)
    (cache / "train.json").write_text(json.dumps(metadata, indent=2) + "\n")
    inference = [{"video_path": str(video), "kps_path": str(weighted), "mode": "test"}]
    (cache / "inference.json").write_text(json.dumps(inference, indent=2) + "\n")
