# Video Motion Graphs

[Project](https://h-liu1997.github.io/Video-Motion-Graphs/) · [Paper](https://arxiv.org/abs/2503.20218) · [Models and examples](https://huggingface.co/H-Liu1997/Video-Motion-Graphs)

Music-driven video motion graphs with pose-guided diffusion interpolation.
This release includes graph construction and search, raw-video preprocessing,
the original interpolation checkpoint, and its weighted-pose training code.

## Installation

Install `uv` first, then run every command below from the repository root. The verified environment is
Linux x86_64, Python 3.10, PyTorch 2.2.2 / CUDA 12.1 and one NVIDIA H100 80GB.

```sh
git clone https://github.com/H-Liu1997/Video-Motion-Graphs.git
cd Video-Motion-Graphs
uv venv --python 3.10 .venv
source .venv/bin/activate
uv pip install --python .venv/bin/python --no-deps --index-strategy unsafe-best-match --extra-index-url https://download.pytorch.org/whl/cu121 -r requirements.lock.txt
python scripts/download_assets.py
```

The lock preserves headless OpenCV and the matching CUDA 12 ONNX Runtime wheel.
Video encoding uses PyAV and the FFmpeg binary bundled with imageio-ffmpeg.
The downloader checks SHA256 for every file and skips valid existing files.
`asset_manifest.json` pins upstream model revisions and the original VMG
checkpoint. The downloads use about 16 GB; no Drive authentication is needed.

## Generate a dance video

```sh
python vmg/train.py -c vmg/configs/demo.yaml
```

The supplied example starts with `examples/Sequence_01_6.mp4` and
`examples/beat_search_2_audio.mp3`. It extracts pose and segmentation, builds a
motion graph, searches a music-conditioned path, generates diffusion transition
frames and writes the video with audio under `outputs/vmg/random_*/res_0_audio.mp4`.
The entrypoint retains its historical name `train.py`; this command performs
graph construction and generation, not diffusion model training.

Intermediate files are stored under `cache/Sequence_01_6/`. The first run needs
no precomputed poses, graph distances or generated transition clips. Later runs
reuse that cache. Use a distinct cache directory for each source video.

For another video and music track:

```sh
python vmg/train.py -c vmg/configs/demo.yaml --data_path /path/to/videos/ --video_ori_name my_video --audio_path /path/to/music.mp3 --cache_path cache/my_video/ --out_path outputs/my_video/
```

The input filename is `my_video.mp4`. This example is designed for a single
visible dancer and depends on suitable repeated poses in the source footage.
The supplied search configuration uses the demo's beat spacing; adapting the
search to another track may require changing its rhythm parameters in
`vmg/vmgv10_trainer.py`.

## Train and resume interpolation

Create the pose videos, weighted heatmaps, hand/face masks and training metadata:

```sh
python vmg/train.py -c vmg/configs/demo.yaml --preprocess_only
```

Then run two real parameter updates from the released model, save the complete
training state, and start a new process that resumes through update three:

```sh
python diffusion/train.py --config diffusion/configs/train.yaml
python diffusion/train.py --config diffusion/configs/resume.yaml
```

Checkpoints are written to `outputs/diffusion/train/checkpoint-2` and
`checkpoint-3`. They include model, optimizer, scheduler, mixed-precision scaler
and RNG state. The pair needs about 26 GB of additional disk space. W&B logging
is disabled by default; set `WANDB_MODE=online` to enable it with your account.

The demo configs use batch size 1, 14 frames and 512×512 training crops. They
validate save/resume on real data, not convergence or a held-out benchmark.
Train and validation metadata use the same demo video. For a full training run,
supply separate train/test videos in the metadata and change
`solver.max_train_steps` and validation intervals in the YAML.

Each training metadata row has `video_path`, `kps_path`, `mask_path` and `mode`
(`train` or `test`). Paths are relative to the repository root or absolute.
`kps_path` names the unweighted pose video; the training loader reads its sibling
`*_weighted_heatmap.mp4`. The preprocessing command writes the matching schema
to `cache/Sequence_01_6/train.json`.

For standalone interpolation after preprocessing:

```sh
python diffusion/inference.py --config diffusion/configs/inference.yaml
```

The original checkpoint uses a four-channel seed guider and 14 frames: three
context frames at each end and eight generated middle frames. The supplied
inference config uses 20 denoising steps and guidance scale 3.5. Do not substitute
a five-channel model configuration for this checkpoint.

## Verification and scope

The release is validated with real source video and music on one H100. Short
training checks verify parameter updates and full-state recovery. Multi-GPU
training, long-run convergence and the complete paper benchmark are not covered
by these checks.

The raw example run processed 3,051 frames, built a 762-node / 1,059-edge graph,
generated three new transitions and produced an 18-second, 768×768, 30fps video
with audio in 12m02s. The two training commands completed in 3m10s.
[View the reproduced video](https://huggingface.co/H-Liu1997/Video-Motion-Graphs/resolve/main/examples/generated_demo.mp4).

## License

Original project code is Apache-2.0; see `LICENSE`. Adapted third-party source
retains its original terms and attribution in `NOTICE` and `licenses/`.
The released diffusion weights retain CreativeML OpenRAIL-M from their Stable
Diffusion base. Other downloaded models retain their upstream terms. Example
media is supplied for reproducing the demonstration and is not covered by the
source-code license.
