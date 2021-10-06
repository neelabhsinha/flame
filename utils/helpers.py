import os.path
import pickle

import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from joblib import Parallel, delayed
from scipy.stats import multivariate_normal
import random

from config import mpiigaze_processed_data


# Generating Probability Distribution Heatmap

def points_to_gaussian_heatmap(centers, height, width, covariance):
    """
    obtains probability distribution heatmap (one channel) for a given set of points
    :param centers: set of landmark points
    :param height: height of the image
    :param width: weidth of the image
    :param covariance: covariance matrix
    :return:
    """
    gaussians = []
    for x, y in centers:
        s = np.eye(2) * covariance
        g = multivariate_normal(mean=(x, y), cov=s)
        gaussians.append(g)
    x = np.arange(0, width)
    y = np.arange(0, height)
    xx, yy = np.meshgrid(x, y)
    xxyy = np.stack([xx.ravel(), yy.ravel()]).T
    dist = sum(g.pdf(xxyy) for g in gaussians)
    img = dist.reshape((height, width))
    return img


def get_2d_heatmap(fl_2d, height, width, covariance):
    """
    obtains 28 channel heatmap using the above function
    :param fl_2d: 2D landmark points
    :param height: height of the required heatmap image
    :param width: width of the required heatmap image
    :param covariance: covariance matrix
    :return: multi-channel heatmap with one channel corresponding to one eye landmark point
    """
    channels = []
    for i in range(0, 28):
        landmark_right = (fl_2d[i], fl_2d[i + 56])
        landmark_left = (fl_2d[i + 28], fl_2d[i + 56 + 28])
        centers = [landmark_right, landmark_left]
        channels.append(points_to_gaussian_heatmap(centers, height, width, covariance))
    channels = np.array(channels)
    channels = np.swapaxes(channels, 0, 2)
    channels = np.swapaxes(channels, 0, 1)
    return channels


def get_eye_and_heatmap(frame, heatmap, fl_2d, resolution, split, idx, dataset):
    """
    randomly crop an eye from the complete face image (see data.py for significance of this method)
    """
    offset = (0, 0)
    if dataset == 'mpiigaze':
        left_center = ((fl_2d[36] + fl_2d[42]) / 2, (fl_2d[39 + 56] + fl_2d[45 + 56]) / 2)
        right_center = ((fl_2d[8] + fl_2d[14]) / 2, (fl_2d[11 + 56] + fl_2d[17 + 56]) / 2)
        offset = (max(int(right_center[0]) - 130, 0), max(0, min(int(left_center[1]), int(right_center[1])) - 130))
    if split == 'train':
        left_eye = random.randint(0, 1)
    else:
        left_eye = idx % 2
    if left_eye == 1:
        center = ((fl_2d[36] + fl_2d[42]) / 2 - offset[0], (fl_2d[39 + 56] + fl_2d[45 + 56]) / 2 - offset[1])
        if center[0] + 60 < frame.shape[1]:
            frame = frame[int(center[1]) - 60:int(center[1]) + 60, int(center[0]) - 60:int(center[0]) + 60]
            heatmap = heatmap[int(center[1]) - 60:int(center[1]) + 60, int(center[0]) - 60:int(center[0]) + 60]
        else:
            frame = frame[int(center[1]) - 60:int(center[1]) + 60, (frame.shape[1] - 120):frame.shape[1]]
            heatmap = heatmap[int(center[1]) - 60:int(center[1]) + 60, (frame.shape[1] - 120):frame.shape[1]]
        fl_2d_eye = np.concatenate((fl_2d[0:28], fl_2d[56:56 + 28]))
    else:
        center = ((fl_2d[8] + fl_2d[14]) / 2 - offset[0], (fl_2d[11 + 56] + fl_2d[17 + 56]) / 2 - offset[1])
        if center[0] - 60 >= 0:
            frame = frame[int(center[1]) - 60:int(center[1]) + 60, int(center[0]) - 60:int(center[0]) + 60]
            heatmap = heatmap[int(center[1]) - 60:int(center[1]) + 60, int(center[0]) - 60:int(center[0]) + 60]
        else:
            frame = frame[int(center[1]) - 60:int(center[1]) + 60, 0: 120]
            heatmap = heatmap[int(center[1]) - 60:int(center[1]) + 60, 0: 120]
        fl_2d_eye = np.concatenate((fl_2d[0 + 28:28 + 28], fl_2d[56 + 28:56 + 28 + 28]))

    if (frame.shape[0] > 0 and frame.shape[0] < 120) or (frame.shape[1] > 0 and frame.shape[1] < 120):
        cv2.resize(frame, (120, 120))
        cv2.resize(heatmap, (120, 120))
    elif frame.shape[0] == 0 or frame.shape[1] == 0:
        frame = np.zeros((120, 120, 3))
        heatmap = np.zeros((120, 120, 28))
    if resolution != 120:
        frame = cv2.resize(frame, (resolution, resolution))
        heatmap = cv2.resize(heatmap, (resolution, resolution))
    return frame, heatmap, fl_2d_eye


# GCN Helpers

def get_adjacency_matrix():
    """
    get adjacency matrix of the eye
    :return: adj: adjacency matrix of the eye structure based on openface 2.0
    """
    edges = [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 0], [8, 9], [9, 10], [10, 11], [11, 12],
             [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 8], [20, 21], [21, 22],
             [22, 23], [23, 24], [24, 25], [25, 26], [26, 27], [27, 20]]
    size = len(set([n for e in edges for n in e]))
    adj = [[0] * size for _ in range(size)]
    for sink, source in edges:
        adj[sink][source] = 1
        adj[source][sink] = 1
    adj = np.array(adj) + np.eye(28, 28)
    norm = np.linalg.norm(adj)
    adj = adj / norm
    adj = torch.from_numpy(adj)
    return adj
