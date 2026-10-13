import os
import sys
import time
import random
import json
import subprocess
from pathlib import Path

import av
import cv2
import igraph as ig
import matplotlib
import numpy as np
import torch
from controlnet_aux.util import HWC3
from decord import VideoReader
from joblib import Parallel, delayed
from loguru import logger
from moviepy.editor import ImageSequenceClip
from PIL import Image
from tqdm import tqdm

from dataloaders.vmgv3 import CustomDataset
from utils.genextend_inference_utils import adjust_statistics_to_match_reference

import math

eps = 1e-5

def read_frames_v2(video_path, size=512):
    container = av.open(video_path)

    video_stream = next(s for s in container.streams if s.type == "video")
    frames = []
    for packet in container.demux(video_stream):
        for frame in packet.decode():
            image = Image.frombytes(
                "RGB",
                (frame.width, frame.height),
                frame.to_rgb().to_ndarray(),
            )
            frames.append(np.array(image)[:, :, :])

    return frames

def draw_bodypose(canvas, candidate, subset):
    H, W, C = canvas.shape
    candidate = np.array(candidate)
    subset = np.array(subset)

    stickwidth = 4

    limbSeq = [
        [2, 3],
        [2, 6],
        [3, 4],
        [4, 5],
        [6, 7],
        [7, 8],
        [2, 9],
        [9, 10],
        [10, 11],
        [2, 12],
        [12, 13],
        [13, 14],
        [2, 1],
        [1, 15],
        [15, 17],
        [1, 16],
        [16, 18],
        [3, 17],
        [6, 18],
    ]

    colors = [
        [255, 0, 0],
        [255, 85, 0],
        [255, 170, 0],
        [255, 255, 0],
        [170, 255, 0],
        [85, 255, 0],
        [0, 255, 0],
        [0, 255, 85],
        [0, 255, 170],
        [0, 255, 255],
        [0, 170, 255],
        [0, 85, 255],
        [0, 0, 255],
        [85, 0, 255],
        [170, 0, 255],
        [255, 0, 255],
        [255, 0, 170],
        [255, 0, 85],
    ]

    for i in range(17):
        for n in range(len(subset)):
            index = subset[n][np.array(limbSeq[i]) - 1]
            if -1 in index:
                continue
            Y = candidate[index.astype(int), 0] * float(W)
            X = candidate[index.astype(int), 1] * float(H)
            mX = np.mean(X)
            mY = np.mean(Y)
            length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
            angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
            polygon = cv2.ellipse2Poly(
                (int(mY), int(mX)), (int(length / 2), stickwidth), int(angle), 0, 360, 1
            )
            cv2.fillConvexPoly(canvas, polygon, colors[i])

    canvas = (canvas * 0.6).astype(np.uint8)
    for i in range(18):
        for n in range(len(subset)):
            index = int(subset[n][i])
            if index == -1:
                continue
            x, y = candidate[index][0:2]
            x = int(x * W)
            y = int(y * H)
            cv2.circle(canvas, (int(x), int(y)), 4, colors[i], thickness=-1)

    return canvas

def draw_handpose(canvas, all_hand_peaks):
    H, W, C = canvas.shape

    edges = [
        [0, 1],
        [1, 2],
        [2, 3],
        [3, 4],
        [0, 5],
        [5, 6],
        [6, 7],
        [7, 8],
        [0, 9],
        [9, 10],
        [10, 11],
        [11, 12],
        [0, 13],
        [13, 14],
        [14, 15],
        [15, 16],
        [0, 17],
        [17, 18],
        [18, 19],
        [19, 20],
    ]

    for peaks in all_hand_peaks:
        peaks = np.array(peaks)

        for ie, e in enumerate(edges):
            x1, y1 = peaks[e[0]]
            x2, y2 = peaks[e[1]]
            x1 = int(x1 * W)
            y1 = int(y1 * H)
            x2 = int(x2 * W)
            y2 = int(y2 * H)
            if x1 > eps and y1 > eps and x2 > eps and y2 > eps:
                cv2.line(
                    canvas,
                    (x1, y1),
                    (x2, y2),
                    matplotlib.colors.hsv_to_rgb([ie / float(len(edges)), 1.0, 1.0])
                    * 255,
                    thickness=2,
                )

        for i, keyponit in enumerate(peaks):
            x, y = keyponit
            x = int(x * W)
            y = int(y * H)
            if x > eps and y > eps:
                cv2.circle(canvas, (x, y), 4, (0, 0, 255), thickness=-1)
    return canvas

def blend_frames(frame_a, frame_b, blend_ratio):
    """
    Blends two frames together based on the blend ratio.

    Args:
    - frame_a: The first frame data (could be body, hand, or face keypoints).
    - frame_b: The second frame data.
    - blend_ratio: A value between 0 and 1 indicating how much of frame_b to blend into frame_a.

    Returns:
    - A blended frame data.
    """
    return (1 - blend_ratio) * frame_a + blend_ratio * frame_b

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

def blending_2d_images_version2(node_op_body, node_op_hand, node_op_face, node_op_body_id, H=720, W=784, save_path=None, rgb_frames=None, seed_frames=8):
    if rgb_frames is not None:
        rgb_save_path = save_path.replace("blend_", "rgb_")
        rgb_frames = [frame for frame in rgb_frames]
        clip = ImageSequenceClip(rgb_frames, fps=30)
        clip.write_videofile(rgb_save_path, codec='libx264', bitrate='4000k')

    total_frames = len(node_op_body)
    print(total_frames)

    blended_op_body = [None] * total_frames
    blended_op_hands = [None] * total_frames
    blended_op_face = [None] * total_frames
    blended_op_body[0+seed_frames], blended_op_body[-1-seed_frames] = node_op_body[0+seed_frames], node_op_body[-1-seed_frames]
    blended_op_hands[0+seed_frames], blended_op_hands[-1-seed_frames] = node_op_hand[0+seed_frames], node_op_hand[-1-seed_frames]
    blended_op_face[0+seed_frames], blended_op_face[-1-seed_frames] = node_op_face[0+seed_frames], node_op_face[-1-seed_frames]

    for i in range(1+seed_frames, total_frames - 1 - seed_frames):
        blend_ratio = (i - seed_frames) / (total_frames - 2 * seed_frames - 1)
        blended_op_body[i] = blend_frames(node_op_body[0+seed_frames], node_op_body[-1-seed_frames], blend_ratio)
        blended_op_hands[i] = blend_frames(node_op_hand[0+seed_frames], node_op_hand[-1-seed_frames], blend_ratio)
        blended_op_face[i] = blend_frames(node_op_face[0+seed_frames], node_op_face[-1-seed_frames], blend_ratio)

    blended_op_body[:seed_frames] = node_op_body[:seed_frames]
    blended_op_body[-seed_frames:] = node_op_body[-seed_frames:]
    blended_op_hands[:seed_frames] = node_op_hand[:seed_frames]
    blended_op_hands[-seed_frames:] = node_op_hand[-seed_frames:]
    blended_op_face[:seed_frames] = node_op_face[:seed_frames]
    blended_op_face[-seed_frames:] = node_op_face[-seed_frames:]

    blended_op_body_frames = np.array(blended_op_body)
    blended_op_hands_frames = np.array(blended_op_hands)
    blended_op_face_frames = np.array(blended_op_face)
    id_body_frames = node_op_body_id
    print(blended_op_body_frames.shape, blended_op_hands_frames.shape, blended_op_face_frames.shape, id_body_frames.shape)

    kps_results = []
    np_results = []
    for candidate, subset, hands, faces in zip(blended_op_body_frames,id_body_frames,  blended_op_hands_frames, blended_op_face_frames):
        canvas = np.zeros(shape=(H, W, 3), dtype=np.uint8)
        canvas = draw_bodypose(canvas, candidate, subset)
        canvas = draw_handpose(canvas, hands)
        detected_map = HWC3(canvas)
        output_type = "pil"
        if output_type == "pil":
            detected_map_pil = Image.fromarray(detected_map)
            kps_results.append(detected_map_pil)
        np_results.append(detected_map)

    save_videos_from_pil(kps_results, save_path, fps=30, bitrate=4000000)
    np_results = np.array(np_results)
    print(np_results.shape)
    return np_results

def compute_distances(
        node_hands_last, node_hands_vel_last, node_body_pos_last, node_body_vel_last, node_face_pos_last, node_face_vel_last,
        p_node_hands_first_list, p_node_hands_vel_first_list, p_node_body_pos_list, p_node_body_vel_list, p_node_face_pos_list, p_node_face_vel_list,
        joints):
    pos_distances = []
    vel_distances = []
    hands_distances = []
    hands_vel_distances = []
    face_pos_distances = []
    face_vel_distances = []

    for p_node_hands, p_node_hands_vel, p_node_body_pos, p_node_body_vel, p_node_face_pos, p_node_face_vel, in zip(p_node_hands_first_list, p_node_hands_vel_first_list, p_node_body_pos_list, p_node_body_vel_list, p_node_face_pos_list, p_node_face_vel_list):

        hands_dis = np.sum(np.linalg.norm(p_node_hands.reshape(42, 2) - node_hands_last.reshape(42, 2), axis=-1))
        hands_vel_dis = np.sum(np.linalg.norm(p_node_hands_vel.reshape(42, 2) - node_hands_vel_last.reshape(42, 2), axis=-1))
        pos_dis = np.sum(np.linalg.norm(p_node_body_pos.reshape(18, 2) - node_body_pos_last.reshape(18, 2), axis=-1))
        vel_dis = np.sum(np.linalg.norm(p_node_body_vel.reshape(18, 2) - node_body_vel_last.reshape(18, 2), axis=-1))
        face_pos_dis = np.sum(np.linalg.norm(p_node_face_pos.reshape(68, 2) - node_face_pos_last.reshape(68, 2), axis=-1))
        face_vel_dis = np.sum(np.linalg.norm(p_node_face_vel.reshape(68, 2) - node_face_vel_last.reshape(68, 2), axis=-1))
        hands_distances.append(hands_dis)
        hands_vel_distances.append(hands_vel_dis)
        pos_distances.append(pos_dis)
        vel_distances.append(vel_dis)
        face_pos_distances.append(face_pos_dis)
        face_vel_distances.append(face_vel_dis)
    return hands_distances, hands_vel_distances, pos_distances, vel_distances, face_pos_distances, face_vel_distances

class CustomTrainer:
    def __init__(self, args):
        self.args = args
        self.checkpoint_path = args.out_path
        self.train_data = CustomDataset(args)
        self.train_loader = torch.utils.data.DataLoader(
            self.train_data, batch_size=1, shuffle=False,
            num_workers=args.loader_workers, drop_last=False,
        )

    def train(self, epoch):
        t_start = time.time()
        new_all = self.args.new_all

        if not os.path.exists(self.args.cache_path + "meta_nodes_fix_pixel.pkl") or new_all:
            os.makedirs(self.args.cache_path, exist_ok=True)
            nodes = ig.Graph(directed=True)
            nodes.add_vertices(len(self.train_loader))
            for real_its, dict_data in tqdm(enumerate(self.train_loader)):
                body_pos = dict_data["pose"]
                body_vel = dict_data["pose_val"]
                face_pos = dict_data["facial"]
                face_vel = dict_data["facial_val"]

                hands_pos = dict_data["hands"]
                hands_vel = dict_data["hands_val"]

                op_body = dict_data["op_body"]
                op_hand = dict_data["op_hand"]
                op_face = dict_data["op_face"]
                op_body_id = dict_data["op_body_id"]

                vid = dict_data["vid"]
                nodes.vs[real_its]['body_pos'] = body_pos[0].cpu().numpy()
                nodes.vs[real_its]['body_vel'] = body_vel[0].cpu().numpy()
                nodes.vs[real_its]['face_pos'] = face_pos[0].cpu().numpy()
                nodes.vs[real_its]['face_vel'] = face_vel[0].cpu().numpy()
                nodes.vs[real_its]['hands'] = hands_pos[0].cpu().numpy()
                nodes.vs[real_its]['hands_vel'] = hands_vel[0].cpu().numpy()

                nodes.vs[real_its]['op_body'] = op_body[0].cpu().numpy()
                nodes.vs[real_its]['op_hand'] = op_hand[0].cpu().numpy()
                nodes.vs[real_its]['op_face'] = op_face[0].cpu().numpy()
                nodes.vs[real_its]['op_body_id'] = op_body_id[0].cpu().numpy()

                nodes.vs[real_its]['data_path'] = vid[-2][0]
                nodes.vs[real_its]['video_name'] = vid[-1][0]
                nodes.vs[real_its]['same_vid'] = vid[3].numpy()
                nodes.vs[real_its]['current_id'] = vid[1].numpy()
                nodes.vs[real_its]['previous_id'] = vid[0].numpy()
                nodes.vs[real_its]['next_id'] = vid[2].numpy()

            for its, node in tqdm(enumerate(nodes.vs)):
                joint_previous = node['body_pos'][-1]
                vel_previous = node['body_vel'][-1]
                face_joint_previous = node['face_pos'][-1]
                face_vel_previous = node['face_vel'][-1]
                hands_joint_previous = node['hands'][-1]
                hands_vel_previous = node['hands_vel'][-1]

                if node['next_id'] == -1:
                    joint_next_1 = node['body_pos'][-2]
                    joint_next_2 = node['body_pos'][0]
                    vel_next_1 = node['body_vel'][-2]
                    vel_next_2 = node['body_vel'][0]
                    hands_next_1 = node['hands'][-2]
                    hands_next_2 = node['hands'][0]
                    hands_vel_next_1 = node['hands_vel'][-2]
                    hands_vel_next_2 = node['hands_vel'][0]

                    face_joint_next_1 = node['face_pos'][-2]
                    face_joint_next_2 = node['face_pos'][0]
                    face_vel_next_1 = node['face_vel'][-2]
                    face_vel_next_2 = node['face_vel'][0]

                else:
                    joint_next_1 = nodes.vs[its+1]['body_pos'][0]
                    joint_next_2 = nodes.vs[its+1]['body_pos'][-1]
                    vel_next_1 = nodes.vs[its+1]['body_vel'][0]
                    vel_next_2 = nodes.vs[its+1]['body_vel'][-1]
                    hands_next_1 = nodes.vs[its+1]['hands'][0]
                    hands_next_2 = nodes.vs[its+1]['hands'][-1]
                    hands_vel_next_1 = nodes.vs[its+1]['hands_vel'][0]
                    hands_vel_next_2 = nodes.vs[its+1]['hands_vel'][-1]

                    face_joint_next_1 = nodes.vs[its+1]['face_pos'][0]
                    face_joint_next_2 = nodes.vs[its+1]['face_pos'][-1]
                    face_vel_next_1 = nodes.vs[its+1]['face_vel'][0]
                    face_vel_next_2 = nodes.vs[its+1]['face_vel'][-1]

                joint_distance = np.sum(np.linalg.norm(joint_next_1.reshape(18, 2) - joint_previous.reshape(18, 2), axis=-1))
                vel_distance = np.sum(np.linalg.norm(vel_next_1.reshape(18, 2) - vel_previous.reshape(18, 2), axis=-1))
                face_joint_distance = np.sum(np.linalg.norm(face_joint_next_1.reshape(68, 2) - face_joint_previous.reshape(68, 2), axis=-1))
                face_vel_distance = np.sum(np.linalg.norm(face_vel_next_1.reshape(68, 2) - face_vel_previous.reshape(68, 2), axis=-1))
                hands_joint_distance = np.sum(np.linalg.norm(hands_next_1.reshape(42, 2) - hands_joint_previous.reshape(42, 2), axis=-1))
                hands_vel_distance = np.sum(np.linalg.norm(hands_vel_next_1.reshape(42, 2) - hands_vel_previous.reshape(42, 2), axis=-1))

                nodes.vs[its]['position_threshold'] = joint_distance
                nodes.vs[its]['velocity_threshold'] = vel_distance
                nodes.vs[its]['hands_threshold'] = hands_joint_distance
                nodes.vs[its]['hands_velocity_threshold'] = hands_vel_distance
                nodes.vs[its]['face_position_threshold'] = face_joint_distance
                nodes.vs[its]['face_velocity_threshold'] = face_vel_distance

            nodes.write_pickle(fname=self.args.cache_path + "meta_nodes_fix_pixel.pkl")
        else:
            nodes = ig.Graph.Read_Pickle(fname=self.args.cache_path + "meta_nodes_fix_pixel.pkl")

        bad_edges = []
        if not os.path.exists(self.args.cache_path + "hands_distances_pixel_1frame.npy") or new_all:
            hands_0 = np.array([node['hands'][0] for node in nodes.vs])
            hands_vel_0 = np.array([node['hands_vel'][0] for node in nodes.vs])
            body_pos_0 = np.array([node['body_pos'][0] for node in nodes.vs])
            body_vel_0 = np.array([node['body_vel'][0] for node in nodes.vs])
            face_pos_0 = np.array([node['face_pos'][0] for node in nodes.vs])
            face_vel_0 = np.array([node['face_vel'][0] for node in nodes.vs])

            pos_distances = []
            vel_distances = []
            hands_distances = []
            hands_vel_distances = []
            face_pos_distances = []
            face_vel_distances = []
            pixel_distances = []

            args = [
                (node['hands'][-1],
                 node['hands_vel'][-1],
                node['body_pos'][-1],
                node['body_vel'][-1],
                node['face_pos'][-1],
                node['face_vel'][-1],
                hands_0,
                hands_vel_0,
                body_pos_0,
                body_vel_0,
                face_pos_0,
                face_vel_0,
                18) for node in nodes.vs]
            results = Parallel(n_jobs=-1)(delayed(compute_distances)(*i) for i in tqdm(args))
            for res in results:
                hands_distances.extend(res[0])
                hands_vel_distances.extend(res[1])
                pos_distances.extend(res[2])
                vel_distances.extend(res[3])
                face_pos_distances.extend(res[4])
                face_vel_distances.extend(res[5])
            np.save(self.args.cache_path + 'pos_distances_pixel_1frame.npy', pos_distances)
            np.save(self.args.cache_path + 'vel_distances_pixel_1frame.npy', vel_distances)
            np.save(self.args.cache_path + 'hands_distances_pixel_1frame.npy', hands_distances)
            np.save(self.args.cache_path + 'hands_vel_distances_1frame.npy', hands_vel_distances)
            np.save(self.args.cache_path + 'face_pos_distances_pixel_1frame.npy', face_pos_distances)
            np.save(self.args.cache_path + 'face_vel_distances_pixel_1frame.npy', face_vel_distances)
        else:
            pos_distances = np.load(self.args.cache_path + 'pos_distances_pixel_1frame.npy')
            vel_distances = np.load(self.args.cache_path + 'vel_distances_pixel_1frame.npy')
            hands_distances = np.load(self.args.cache_path + 'hands_distances_pixel_1frame.npy')
            hands_vel_distances = np.load(self.args.cache_path + 'hands_vel_distances_1frame.npy')
            face_pos_distances = np.load(self.args.cache_path + 'face_pos_distances_pixel_1frame.npy')
            face_vel_distances = np.load(self.args.cache_path + 'face_vel_distances_pixel_1frame.npy')

        for i, node in tqdm(enumerate(nodes.vs)):
            if node['next_id'] != -1 and nodes.vs[i]["same_vid"] == nodes.vs[i+1]["same_vid"]:
                nodes.add_edge(i, i+1, p_d = pos_distances[i*len(nodes.vs)+i+1], v_d = vel_distances[i*len(nodes.vs)+i+1], iou_d = hands_distances[i*len(nodes.vs)+i+1], original = 1)
            for j, p_node in enumerate(nodes.vs):
                if node['next_id'] != -1 and j == i+1: continue
                if i == j: continue
                if j + 1 == i: continue
                if float(vel_distances[i*len(nodes.vs)+j]) < node["velocity_threshold"]*self.args.threshold_list[0]\
                    and float(pos_distances[i*len(nodes.vs)+j]) < node["position_threshold"]*self.args.threshold_list[0]\
                    and float(face_pos_distances[i*len(nodes.vs)+j]) < node["face_position_threshold"]*self.args.threshold_list[1]\
                    and float(face_vel_distances[i*len(nodes.vs)+j]) < node["face_velocity_threshold"]*self.args.threshold_list[1]\
                    and float(hands_distances[i*len(nodes.vs)+j]) < node["hands_threshold"]*self.args.threshold_list[2]\
                    and float(hands_vel_distances[i*len(nodes.vs)+j]) < node["hands_velocity_threshold"]*self.args.threshold_list[2]:
                    nodes.add_edge(
                        i, j, p_d = pos_distances[i*len(nodes.vs)+j], v_d = vel_distances[i*len(nodes.vs)+j],
                        f_p_d = face_pos_distances[i*len(nodes.vs)+j], f_v_d = face_vel_distances[i*len(nodes.vs)+j],
                        h_d = hands_distances[i*len(nodes.vs)+j], h_v_d=hands_vel_distances[i*len(nodes.vs)+j], original = 0)
        for edge in bad_edges:
            if nodes.are_connected(edge[0], edge[1]):
                nodes.delete_edges([(edge[0], edge[1])])
                logger.info(f"delete edge {edge[0]} {edge[1]}")
        nodes.write_pickle(fname=self.args.cache_path + f"reduced_nodes_{self.args.threshold_list[0]}_{self.args.threshold_list[1]}_{self.args.threshold_list[2]}.pkl")
        logger.info(f"nodes: {len(nodes.vs)}, edges: {len(nodes.es)}")
        in_degrees = nodes.indegree()
        out_degrees = nodes.outdegree()
        avg_in_degree = sum(in_degrees) / len(in_degrees)
        avg_out_degree = sum(out_degrees) / len(out_degrees)
        logger.info(f"Average In-degree: {avg_in_degree}")
        logger.info(f"Average Out-degree: {avg_out_degree}")
        logger.info(f"max in degree: {max(in_degrees)}, max out degree: {max(out_degrees)}")
        logger.info(f"min in degree: {min(in_degrees)}, min out degree: {min(out_degrees)}")

        ascc = nodes.clusters(mode="STRONG")
        lascc = ascc.giant()

        logger.info(f"original graph: {len(nodes.vs)}, {len(nodes.es)}, lascc: {len(lascc.vs)}, {len(lascc.es)}")
        with open(os.path.join(self.args.out_path, "all_search_results.txt"), "a+") as wfile:
            wfile.write(f"{self.checkpoint_path}    {self.args.threshold_list[0]}    {self.args.threshold_list[1]}    {self.args.threshold_list[2]}  {self.args.threshold_list[3]}     original graph: {len(nodes.vs)}, {len(nodes.es)}, lascc: {len(lascc.vs)}, {len(lascc.es)} \n")
        nodes = nodes #lascc
        unique_v = 72
        self.random_sampling(nodes, 0, save_path=self.checkpoint_path + f"random_{unique_v}/", unique_v=unique_v, n=108)

    def checkblending(self, nodes, save_path, transitions):
        os.makedirs(save_path, exist_ok=True)
        seed_frames = 3
        video_reader = VideoReader(nodes.vs[0]["data_path"] + nodes.vs[0]["video_name"] + ".mp4")
        meta_infos = []
        for source_index, target_index in transitions:
            if os.path.exists(os.path.join(save_path, f"res_blend_res_{source_index}_{target_index}.mp4")):
                continue
            first_clip_start = source_index * self.args.pose_length - seed_frames
            first_clip_end = (source_index + 1) * self.args.pose_length
            second_clip_start = target_index * self.args.pose_length
            second_clip_end = (target_index + 1) * self.args.pose_length + seed_frames
            frame_indices = list(range(first_clip_start, first_clip_end)) + list(range(second_clip_start, second_clip_end))
            all_frames = video_reader.get_batch(frame_indices).asnumpy()
            pose_inputs = [
                np.concatenate([
                    nodes.vs[source_index-1][key][-seed_frames:],
                    nodes.vs[source_index][key],
                    nodes.vs[target_index][key],
                    nodes.vs[target_index+1][key][:seed_frames],
                ], axis=0)
                for key in ("op_body", "op_hand", "op_face", "op_body_id")
            ]
            blending_2d_images_version2(
                *pose_inputs,
                save_path=os.path.join(save_path, f"res_blend_{source_index}_{target_index}.mp4"),
                rgb_frames=all_frames,
                H=all_frames.shape[1], W=all_frames.shape[2], seed_frames=seed_frames,
            )
            meta_infos.append({
                "kps_path": os.path.join(save_path, f"res_blend_{source_index}_{target_index}.mp4"),
                "video_path": os.path.join(save_path, f"res_rgb_{source_index}_{target_index}.mp4"),
                "mode": "test",
            })
        json_path = os.path.join(save_path, "blending_meta.json")
        json.dump(meta_infos, open(json_path, "w"))
        if meta_infos:
            subprocess.run(
                [sys.executable, os.path.join(self.args.pose2video_path, "inference.py"), "--config", self.args.interpolation_config,
                 "--json", os.path.abspath(json_path)],
                check=True,
            )

    def random_sampling(self, nodes, global_beta, save_path, unique_v=50, n=110, k=12, s=0.20):
        def check_proximity(idx):
            idx_4 = idx * 4
            n = round((idx_4 - 5) / 11.2)
            closest_value = 5 + 11.2 * n
            difference = closest_value - idx_4

            if 0 <= difference <= 3:
                return True
            return False

        def check_proximity_gt(idx):
            idx_4 = idx * 4
            n = round(idx_4 / 11.2)
            closest_value = 11.2 * n
            difference = closest_value - idx_4

            if 0 <= difference <= 3:
                return True
            return False

        success_sample = 0
        fail_sample = 0
        sample_num = self.args.sample_count
        average_translation = 0
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        for _ in range(sample_num):
            attempts = 0
            max_attempts = 100000
            while attempts < max_attempts:
                stop_flag = False
                node_index = []
                node_video_path = []
                is_original = []
                node_3d_value = []
                node_frame_id = []
                node_op_body = []
                node_op_hand = []
                node_op_face = []
                node_op_body_id = []
                current_vertex = nodes.vs[1] # random.choice(nodes.vs) #nodes.vs[1]
                last_vertex = current_vertex

                smooth_c = 0
                fail_beat = 0
                for i in range(n):
                    node_index.append(current_vertex.index)
                    node_video_path.append(current_vertex["data_path"] + current_vertex["video_name"] + ".mp4")
                    node_3d_value.append(current_vertex["op_body"])
                    node_frame_id.append(current_vertex["current_id"])
                    node_op_body.append(current_vertex["op_body"])
                    node_op_hand.append(current_vertex["op_hand"])
                    node_op_face.append(current_vertex["op_face"])
                    node_op_body_id.append(current_vertex["op_body_id"])

                    if i == 0:
                        pass
                    else:
                        if nodes.es[nodes.get_eid(last_vertex.index, current_vertex.index)]["original"] == 1:
                            is_original.append([1, -1, -1])
                        else:
                            if i <= 2 or i >= (n-2) or len(is_original) == 0 or is_original[-1] == 0:
                                stop_flag = True
                            else:
                                is_original.append([0, last_vertex.index, current_vertex.index])
                                smooth_c += 1

                    if check_proximity_gt(i):
                        if not check_proximity(current_vertex.index):
                            fail_beat += 1

                    if fail_beat > 10:
                        break

                    if stop_flag:
                        break

                    neighbors = [v for v in nodes.successors(current_vertex) if v not in node_index[-k:]]
                    if not neighbors:
                        break
                    last_vertex = current_vertex
                    current_vertex = nodes.vs[random.choice(neighbors)]

                is_original.append([1, -1, -1])
                if not 1 <= smooth_c <= 4:
                    fail_sample += 1
                    attempts += 1
                    continue
                if any(node not in node_frame_id[n-18:n-2] for node in self.args.end_nodes):
                    attempts += 1
                    continue
                if not stop_flag and len(set(node_index)) >= unique_v and smooth_c/n < s and len(node_index) == n:
                    success_sample += 1
                    logger.info(f"ratio: {success_sample/(success_sample+fail_sample)} smooth ratio: {smooth_c/n} number of smooth: {smooth_c}")
                    logger.info(node_index)
                    break
                else:
                    fail_sample += 1
                    attempts += 1
                    stop_flag = False

            if attempts == max_attempts:
                logger.info(f"fail to sample {attempts} times")
                raise RuntimeError("No valid graph path found in 100000 attempts")

            node_index = np.array(node_index)
            is_original = np.array(is_original)
            node_frame_id = np.array(node_frame_id)
            node_3d_value = np.concatenate(node_3d_value, axis=0)
            node_op_body = np.concatenate(node_op_body, axis=0)
            node_op_hand = np.concatenate(node_op_hand, axis=0)
            node_op_face = np.concatenate(node_op_face, axis=0)
            node_op_body_id = np.concatenate(node_op_body_id, axis=0)
            node_video_path = node_video_path
            if not self.args.use_cached_blends:
                blend_path = self.args.cache_path + f"blended_{self.args.threshold_list[0]}_{self.args.threshold_list[1]}_{self.args.threshold_list[2]}/"
                transitions = [(source, target) for original, source, target in is_original if not original]
                self.checkblending(nodes, blend_path, transitions)
            average_translation = self.blending(
                node_index, is_original, node_3d_value, node_video_path, node_frame_id,
                save_path=save_path, global_beta=global_beta, pre_fix=success_sample-1,
                average_translation=average_translation,
                node_op_body=node_op_body, node_op_hand=node_op_hand, node_op_face=node_op_face, node_op_body_id=node_op_body_id)
        logger.info(f"average_translation: {average_translation/sample_num}")

    def concatenate_clips(
            self, is_original, node_video_path, node_frame_id, save_path, video_fps=30, aduio=None,
            node_op_body=None, node_op_hand=None, node_op_face=None, node_op_body_id=None, trans_number=0.0,
            ):
        from moviepy.editor import VideoFileClip, concatenate_videoclips
        from typing import List, Tuple
        import moviepy.editor as mp
        import cv2
        import numpy as np
        clips = []
        all_frames = []
        for i, (filepath, node_frame) in enumerate(zip(node_video_path, node_frame_id)):
            start_frame = node_frame*self.args.pose_length
            end_frame = node_frame*self.args.pose_length+self.args.pose_length
            cap = cv2.VideoCapture(filepath)
            fps = 30
            start_frame = int(start_frame)
            end_frame = int(end_frame)
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            frames = []
            for cframe in range(end_frame - start_frame):
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame[:,:,::-1])
            all_frames.extend(frames)
            cap.release()

        w, h, c = all_frames[0].shape
        max_side = max(w, h)
        ratio = max_side/768
        if w > h:
            new_w = 768
            new_h = int(h/ratio)
        else:
            new_h = 768
            new_w = int(w/ratio)

        all_frames_resized = []
        for frame in all_frames:
            frame_resized = cv2.resize(frame, (new_h, new_w))
            all_frames_resized.append(frame_resized)
        all_frames = all_frames_resized

        seed_frames = 3
        for i, (_, _, is_original_sub) in enumerate(zip(node_video_path, node_frame_id, is_original)):
            if is_original_sub[0] == 0:
                if i*self.args.pose_length+2*self.args.pose_length+seed_frames > len(all_frames) or i*self.args.pose_length-seed_frames<0:
                    continue
                save_blended_path = self.args.cache_path + f"blended_{self.args.threshold_list[0]}_{self.args.threshold_list[1]}_{self.args.threshold_list[2]}/"
                save_name = save_blended_path + f"res_blend_res_{is_original_sub[1]}_{is_original_sub[2]}.mp4"
                print(save_name)
                synthesized_frames = read_frames_v2(save_name)
                print(len(synthesized_frames))
                start_index = i*self.args.pose_length
                end_index = start_index + 2*self.args.pose_length
                all_frames[start_index:end_index] = synthesized_frames[seed_frames:-seed_frames]

        clip = ImageSequenceClip(all_frames, fps=24)
        clip.write_videofile(save_path, codec='libx264', bitrate="4000k")
        video_reader = VideoReader(save_path)
        video_np = []
        for i in range(len(video_reader)):
            video_frame = video_reader[i].asnumpy()
            video_np.append(Image.fromarray(video_frame))
        adjusted_video_pil = adjust_statistics_to_match_reference([video_np])
        save_videos_from_pil(adjusted_video_pil[0], save_path.replace(".mp4", "_color_fix.mp4"), fps=24, bitrate=4000000)

        aduio = self.args.audio_path
        if aduio is not None:
            audio = mp.AudioFileClip(aduio)
            video = mp.VideoFileClip(save_path.replace(".mp4", "_color_fix.mp4"))
            video = video.set_fps(fps)
            if audio.duration > video.duration:
                audio = audio.subclip(0, video.duration)
            video = video.set_audio(audio)
            video.write_videofile(save_path.replace(".mp4", "_audio.mp4"), fps=fps, codec='libx264', audio_codec='aac', audio=True, bitrate="4000k")
            video.write_videofile(save_path.replace(".mp4", "_audio_low.mp4"), fps=fps, codec='libx264', audio_codec='aac', audio=True, bitrate="400k")

    def blending(
            self, node_index, is_original, node_3d_value, node_video_path, node_frame_id,
            save_path="res", global_beta=None, pre_fix="", average_translation=0, audio=None,
            node_op_body=None, node_op_hand=None, node_op_face=None, node_op_body_id=None,trans_number=0.0,
            ):
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        self.concatenate_clips(
            is_original, node_video_path, node_frame_id, save_path + f"res_{pre_fix}.mp4", aduio=audio,
            node_op_body=node_op_body, node_op_hand=node_op_hand, node_op_face=node_op_face, node_op_body_id=node_op_body_id,trans_number=trans_number,
            )
        return average_translation
