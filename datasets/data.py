import random
import torch
import pickle
from math import ceil
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data.sampler import BatchSampler
import numpy as np
import os
from random import shuffle
from config import project_path
from utils.helpers import get_eye_and_heatmap


# Build index set for sampler
def get_frame_idxs(dataset, vid_paths, frame_window, split):
    idxs = []
    for i, vid_path in enumerate(vid_paths):
        if os.path.splitext(os.path.split(vid_path)[-1])[0] in split:
            with open(vid_path, 'rb') as f:
                vid = pickle.load(f)
                framecount = vid.shape[0]
                num_segments = ceil(framecount / frame_window)
                for j in range(0, framecount - frame_window + 1, frame_window):
                    idxs.append((i, j))  # (i,j) are 0-indexed
    return idxs


# Batch Sampler to generate indexes
class Sampler(BatchSampler):
    def __init__(self, dataset, path, batch_size, frame_window, split, split_nature, divide_part=1):
        self.dataset = dataset
        self.path = path
        self.batch_size = batch_size
        self.frame_window = frame_window
        self.gaze_paths = []
        with open(
                os.path.join(project_path, 'metadata', 'splits', 'data_split_' + dataset + '_' + split_nature + '.pkl'),
                'rb') as f:
            self.data_split = pickle.load(f)
            f.close()
        for gz in sorted(os.listdir(os.path.join(self.path, 'gaze_angles'))):
            if gz.endswith('.pkl'):
                self.gaze_paths.append(os.path.join(self.path, 'gaze_angles', gz))

        self.idxs = get_frame_idxs(self.dataset, self.gaze_paths, self.frame_window, self.data_split[split])
        shuffle(self.idxs)
        self.len_idxs = int(len(self.idxs) * divide_part)
        self.idxs = self.idxs[0:self.len_idxs]

    def __len__(self):
        return self.len_idxs // self.batch_size

    def __iter__(self):
        # for start in range(0, self.len_idxs, self.batch_size):
        #   yield self.idxs[start: start + self.batch_size]

        for start in range(0, self.len_idxs):
            yield self.idxs[start]


# Data class to load samples
class Data(Dataset):
    def __init__(self, name, path, frame_window, split, resolution, split_nature='cross-person', crop='face'):
        self.split = split
        self.resolution = resolution
        self.name = name
        self.crop = crop
        self.path = path
        self.frame_window = frame_window
        self.split_nature = split_nature
        self.vid_paths = []
        self.fl_paths = []
        self.fl_2d_paths = []
        self.hp_paths = []
        self.gaze_paths = []
        self.heatmap_paths = []
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.videomode = True if frame_window > 1 else False
        for vid in sorted(os.listdir(os.path.join(self.path, 'images'))):
            self.vid_paths.append(os.path.join(self.path, 'images', vid))
        for vid in sorted(os.listdir(os.path.join(self.path, 'heatmaps'))):
            self.heatmap_paths.append(os.path.join(self.path, 'heatmaps', vid))
        for fl_2d in sorted(os.listdir(os.path.join(self.path, 'facial_landmarks_2d'))):
            if fl_2d.endswith('.pkl'):
                self.fl_2d_paths.append(os.path.join(self.path, 'facial_landmarks_2d', fl_2d))
        for hp in sorted(os.listdir(os.path.join(self.path, 'head_pose'))):
            if hp.endswith('.pkl'):
                self.hp_paths.append(os.path.join(self.path, 'head_pose', hp))
        for gz in sorted(os.listdir(os.path.join(self.path, 'gaze_angles'))):
            if gz.endswith('.pkl'):
                self.gaze_paths.append(os.path.join(self.path, 'gaze_angles', gz))

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, idx):
        sample = {}
        viddir = self.vid_paths[idx[0]]
        heatmapdir = self.heatmap_paths[idx[0]]
        frames = []
        heatmaps = []
        fl_eye = []
        for k in range(idx[1], idx[1] + self.frame_window):
            with open(self.fl_2d_paths[idx[0]], 'rb') as ffl_2d:
                fl_2ds = pickle.load(ffl_2d)
                fl_2d = fl_2ds[k]
                ffl_2d.close()
            frame = np.load(os.path.join(viddir, str(k + 1) + '.npy'))
            heatmap = np.load(os.path.join(heatmapdir, str(k + 1) + '.npy'))
            if self.crop == 'eye':
                frame, heatmap, fl_2d = get_eye_and_heatmap(frame, heatmap, fl_2d, self.resolution, self.split, k,
                                                            self.name)
            frame = self.transform(frame)
            heatmap = self.transform(heatmap)
            # frame = frame.permute(2,1,0)
            frame = frame.numpy()
            heatmap = heatmap.numpy()
            frames.append(frame)
            fl_eye.append(fl_2d)
            heatmaps.append(heatmap)
        sample["frames"] = np.asarray(frames)
        sample["heatmaps"] = np.asarray(heatmaps)
        sample["fl"] = np.asarray(fl_eye)
        if not self.videomode:
            sample["frames"] = np.squeeze(sample["frames"], axis=0)
            sample["heatmaps"] = np.squeeze(sample["heatmaps"], axis=0)
            sample["fl"] = np.squeeze(sample["fl"], axis=0)
        with open(self.hp_paths[idx[0]], "rb") as fhp:
            hp = pickle.load(fhp)
            sample["hp"] = hp[idx[1]:idx[1] + self.frame_window]
            if not self.videomode:
                sample["hp"] = np.squeeze(sample["hp"], axis=0)
            fhp.close()
        with open(self.gaze_paths[idx[0]], "rb") as fgaze:
            gaze = pickle.load(fgaze)
            sample["gaze"] = gaze[idx[1]:idx[1] + self.frame_window]
            if not self.videomode:
                sample["gaze"] = np.squeeze(sample["gaze"], axis=0)
            fgaze.close()
        return sample
