import numpy as np
import torch
from torch.utils.data import Dataset

from .preprocess import prepare_video


def estimate_linear_velocity(data, dt):
    first = (data[1:2] - data[:1]) / dt
    middle = (data[2:] - data[:-2]) / (2 * dt)
    last = (data[-1:] - data[-2:-1]) / dt
    return np.concatenate([first, middle, last], axis=0)


class CustomDataset(Dataset):
    def __init__(self, args):
        self.args = args
        prepare_video(args)
        prefix = args.preprocessed_path / args.video_ori_name
        self.body = np.load(str(prefix) + "_pose_pos.npy")
        self.body_id = np.load(str(prefix) + "_pose_id.npy")
        self.face = np.load(str(prefix) + "_face_pos.npy")
        self.hands = np.load(str(prefix) + "_hands_pos.npy")
        dt = 1 / args.pose_fps
        self.body_velocity = estimate_linear_velocity(self.body, dt)
        self.face_velocity = estimate_linear_velocity(self.face[:, 0], dt)
        self.hands_velocity = np.concatenate([
            estimate_linear_velocity(self.hands[:, 0], dt),
            estimate_linear_velocity(self.hands[:, 1], dt),
        ], axis=1)

    def __len__(self):
        return len(self.body) // self.args.pose_length

    def __getitem__(self, index):
        start = index * self.args.pose_length
        span = slice(start, start + self.args.pose_length)
        next_id = index + 1 if index + 1 < len(self) else -1
        return {
            "pose": torch.from_numpy(self.body[span]).float(),
            "pose_val": torch.from_numpy(self.body_velocity[span]).float(),
            "facial": torch.from_numpy(self.face[span, 0]).float(),
            "facial_val": torch.from_numpy(self.face_velocity[span]).float(),
            "hands": torch.from_numpy(self.hands[span]).float(),
            "hands_val": torch.from_numpy(self.hands_velocity[span]).float(),
            "op_body": self.body[span],
            "op_body_id": self.body_id[span],
            "op_face": self.face[span],
            "op_hand": self.hands[span],
            "vid": [index - 1, index, next_id, 0, self.args.data_path, self.args.video_ori_name],
        }
