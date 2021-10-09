import os
import logging
import pickle
import sys
from math import pi, atan, sqrt, atan2, asin

import numpy as np

from config import project_path, dataset_paths
from utils.helpers import get_2d_heatmap


def get_max_min(ls):
    """
    get maximum and minimum value of a list
    :param ls: input list
    :return: tensor (max, min)
    """
    if isinstance(ls, list):
        ls = np.asarray(ls)
    return np.max(ls), np.min(ls)


def get_pickle_from_list(ls, destpath, type='list'):
    """
    save a list as pickle
    :param ls: list
    :param destpath: output path
    :param type: type of file (list/np array)
    :return: void
    """
    if type == 'list':
        ls_np = np.asarray(ls)
    else:
        ls_np = ls
    fo = open(destpath, "wb")
    pickle.dump(ls_np, fo, protocol=4)
    fo.close()


def get_head_pose_angles(R):
    """
    obtain (pitch, yaw, roll) of head pose using rotation matrix R
    :param R: rotation matrix
    :return: (pitch, yaw, roll) tensor
    """
    for r in range(0, len(R)):
        R[r] = list(map(float, R[r]))
    if R[2][0] < 1:
        if R[2][0] > -1:
            theta_y = asin(-R[2][0])
            theta_z = atan2(R[1][0], R[0][0])
            theta_x = atan2(R[2][1], R[2][2])
        else:
            theta_y = pi / 2
            theta_z = -atan2(-R[1][2], R[1][1])
            theta_x = 0
    else:
        theta_y = -pi / 2
        theta_z = atan2(-R[1][2], R[1][1])
        theta_x = 0
    return np.array([theta_x, theta_y, theta_z])
    # return np.array([(theta_x), (theta_y), (theta_z)])


def convert_to_degree(radian):
    return float(radian) * 180.0 / pi


def get_angle_from_vector(x, y, z):
    """
    converts a vector (x,y,z) into (yaw,pitch), both provided in WCS
    """
    yaw = (atan(x / z))
    pitch = (atan(y / sqrt(pow(x, 2) + pow(z, 2))))
    gt = np.array([yaw, pitch])
    return gt


def get_mean_and_std(dataset, split_nature='cross-person'):
    """
    obtain dataset metadata for a given dataset
    :param dataset: dataset name
    :param split_nature: nature of split
    :return: void (saves the metadata as pickle)
    """
    path = None
    # Load Data (Add the dataset path in config.py if adding new)
    try:
        path = dataset_paths[dataset]
    except KeyError:
        logging.error('Path to dataset ' + dataset + ' not defined. Please define the same in config.py file')
        sys.exit()
    with open(os.path.join(project_path, 'metadata', 'splits', 'data_split_' + dataset + '_' + split_nature + '.pkl'),
              'rb') as f:
        data_split = pickle.load(f)
        train_videos = data_split['train']
        f.close()
    R_sum = 0
    G_sum = 0
    B_sum = 0
    R_sq_sum = 0
    G_sq_sum = 0
    B_sq_sum = 0
    count = 0
    videos = os.listdir(os.path.join(path, 'images'))
    for vid in videos:
        if vid in train_videos:
            frames = os.listdir(os.path.join(path, 'images', vid))
            for frame in frames:
                with open(os.path.join(path, 'images', vid, frame), 'rb') as fv:
                    img = np.load(fv).astype(float)
                    img = img / 255
                    R_sum += np.sum(img[:, :, 0])
                    G_sum += np.sum(img[:, :, 1])
                    B_sum += np.sum(img[:, :, 2])
                    R_sq_sum += np.sum(img[:, :, 0] ** 2)
                    G_sq_sum += np.sum(img[:, :, 1] ** 2)
                    B_sq_sum += np.sum(img[:, :, 2] ** 2)
                    count += img.shape[0] * img.shape[1]
                    fv.close()

    mean = np.array([float(R_sum) / count, float(G_sum) / count, float(B_sum) / count])
    std = np.sqrt(np.array([((float(R_sq_sum) / count) - (mean[0] ** 2)), ((float(G_sq_sum) / count) - (mean[1] ** 2)),
                            ((float(B_sq_sum) / count) - (mean[2] ** 2))]))
    data_stats = {'mean': mean, 'std': std}
    with open(os.path.join(project_path, 'metadata', 'data_statistics',
                           'data_mean_std_' + dataset + '_' + split_nature + '.pkl'), 'wb') as f:
        pickle.dump(data_stats, f)
        f.close()


def get_and_save_heatmap(dataset):
    """
    get and save heatmap when provided with facial landmarks of the samples
    :param dataset: dataset name
    :return: void
    """
    # Load Data (Add the dataset path in config.py if adding new)
    try:
        destdir = dataset_paths[dataset]
    except KeyError:
        logging.error('Path to dataset ' + dataset + ' not defined. Please define the same in config.py file')
        sys.exit()

    fl_dir = os.path.join(destdir, 'facial_landmarks_2d')
    imgdir = os.path.join(destdir, 'images')
    targetdir = os.path.join(destdir, 'heatmaps')
    if not os.path.exists(targetdir):
        os.mkdir(targetdir)
    files = os.listdir(fl_dir)
    files.sort()
    for file in files:
        pickle_path = os.path.join(fl_dir, file)
        target = os.path.join(targetdir, os.path.splitext(file)[0])
        if not os.path.exists(target):
            os.mkdir(target)
        with open(pickle_path, 'rb') as f:
            fl_2ds = pickle.load(f)
            f.close()
        for (i, fl_2d) in enumerate(fl_2ds):
            get_heatmap_from_idx(i, file, fl_2d, target, imgdir)


def get_heatmap_from_idx(i, file, fl_2d, target, imgdir):
    rgb = np.load(os.path.join(imgdir, os.path.splitext(file)[0], str(i + 1) + '.npy'))
    heatmap = get_2d_heatmap(fl_2d, rgb.shape[0], rgb.shape[1], 1)
    filename = os.path.join(target, str(i + 1) + '.npy')
    np.save(filename, heatmap)
