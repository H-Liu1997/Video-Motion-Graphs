import os
from pathlib import Path

import configargparse


def parse_args():
    parser = configargparse.ArgParser()
    parser.add("-c", "--config", is_config_file=True, required=True)
    parser.add("--data_path", default="examples/")
    parser.add("--video_ori_name", default="Sequence_01_6")
    parser.add("--audio_path", default="examples/beat_search_2_audio.mp3")
    parser.add("--cache_path", default="cache/Sequence_01_6/")
    parser.add("--preprocessed_path", default=None)
    parser.add("--out_path", default="outputs/vmg/")
    parser.add("--pretrained_weights", default="weights/")
    parser.add("--pose2video_path", default="diffusion/")
    parser.add("--interpolation_config", default="diffusion/configs/inference.yaml")
    parser.add("--threshold_list", type=float, nargs=5, default=[3.0, 2.0, 3.0, 1.0, 1.0])
    parser.add("--random_seed", type=int, default=2021)
    parser.add("--sample_count", type=int, default=1)
    parser.add("--loader_workers", type=int, default=0)
    parser.add("--end_nodes", type=int, nargs="*", default=[])
    parser.add("--use_cached_blends", action="store_true")
    parser.add("--new_all", action="store_true")
    parser.add("--preprocess_only", action="store_true")
    args = parser.parse_args()
    for name in ("data_path", "cache_path", "out_path"):
        setattr(args, name, str(Path(getattr(args, name))) + os.sep)
    args.preprocessed_path = Path(args.preprocessed_path or Path(args.cache_path) / "preprocessed")
    args.pose_length = 4
    args.pose_fps = 30
    return args
