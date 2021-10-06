import logging
import math
import os
import pickle
from math import pi

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, BatchSampler

from config import eyediap_processed_data, loggers_loc, test_path, project_path, columbiagaze_processed_data
from datasets.data import Data, Sampler
from losses.angular_loss import AngularGazeLoss
from models.additive_fusion import AdditiveFusionNet
from models.baseline import BaselineNetwork
from models.aggregation_only import ConcatenatedFusionNet
from models.mmtm_fusion import MMTMFusionNet
from utils.helpers import get_adjacency_matrix
from utils.train import forward_propagation


def test_model(frame_window, weight_path, split_nature='cross_person', test_data=None, save_preds=True):
    """
    executes the test of the trained model on a dataset
    :param frame_window: frame_window parameter (with number of frames to process at once
    :param weight_path: complete path of weights of the model on which test run has to be done
    :param split_nature: nature of split
    :param test_data: nature of test dataset if carrying out test on some other dataset for cross-dataset evaluation
    :param save_preds: boolean to provide whether to save all predictions in a dataframe or not
    :return: void (writes a CSV containing the results)
    """
    foldername = os.path.split(os.path.split(weight_path)[0])[-1]
    folderlist = foldername.split('_')
    if len(folderlist) == 8:
        if test_data is None:
            dataset = foldername.split('_')[1]
        else:
            dataset = test_data
        train_dataset = foldername.split('_')[1]
    elif len(folderlist) == 9:
        if test_data is None:
            dataset = folderlist[1]
        else:
            dataset = test_data
        train_dataset = folderlist[2]
    network_name = foldername.split('_')[0]
    network_name_list = network_name.split('-')
    if network_name_list[0] != 'baseline' and len(network_name_list) == 3 or network_name_list[0] == 'baseline' and len(
            network_name_list) == 2:
        resolution = int(network_name_list[-1])
    else:
        resolution = 120
    logging.basicConfig(
        filename=loggers_loc + '/testing_' + train_dataset + '_' + network_name + '_' + split_nature + '.log',
        format='%(asctime)s %(levelname)s : %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p',
        level=logging.INFO)
    logging.info('Starting testing of ' + network_name + ' on ' + dataset + ' for ' + split_nature + ' split')

    # Load Data
    if dataset == 'eyediap':
        path = eyediap_processed_data
    elif dataset == 'columbiagaze':
        path = columbiagaze_processed_data
    test_batch_sampler = BatchSampler(Sampler(dataset, path, 1, frame_window, 'test', split_nature),
                                      1, drop_last=True)
    data_set = Data(dataset, path, frame_window, 'test', resolution=resolution, split_nature=split_nature, crop='eye')

    logging.info('Samplers and Dataset created for ' + dataset)

    # Load Model
    if network_name == 'baseline':  # RGB Baseline Model
        model = BaselineNetwork(in_channels=3)
        learning_rate = 0.00001
    elif network_name == 'additive-fusion':  # F-AF (Additive Fusion Model)
        model = AdditiveFusionNet()
        learning_rate = 0.00005
    elif network_name == 'concatenated-fusion':  # F-AO (FLAME-Aggregation only) model
        model = ConcatenatedFusionNet()
        learning_rate = 0.0001
    elif network_name == 'mmtm-fusion':  # Original FLAME
        model = MMTMFusionNet(input_size=120)
        learning_rate = 0.0001

    # Load weights
    if torch.cuda.is_available():
        cp = torch.load(weight_path, map_location=torch.device('cuda'))
        model.load_state_dict(cp['model_state_dict'])
    else:
        cp = torch.load(weight_path, map_location=torch.device('cpu'))
        model.load_state_dict(cp['model_state_dict'])
    if torch.cuda.is_available():
        model = model.cuda()
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    logging.info(network_name + ' model loaded for testing with weights ' + weight_path)

    # Define Data Loader
    test_loader = DataLoader(data_set, batch_sampler=test_batch_sampler)
    logging.info('DataLoader initialized')
    # Loading Max Min metadata for normalization
    with open(os.path.join(project_path, 'metadata', 'data_statistics', 'data_vals_max_min_' + train_dataset + '.pkl'),
              'rb') as fm:
        data_stats = pickle.load(fm)
        fm.close()
    count = 0

    # Defining Loss Function
    criterion = AngularGazeLoss(data_stats)
    if torch.cuda.is_available():
        criterion = criterion.cuda()
    total_test_loss = 0
    result = []
    adj = None

    # Testing loop
    model.eval()
    for step, data in (enumerate(test_loader)):
        loss_3d, out, y = forward_propagation(model, data, adj, 'eval', data_stats, network_name, criterion)
        out = (torch.squeeze(out) * (data_stats['gaze']['max'] - data_stats['gaze']['min']) + data_stats['gaze'][
            'min']) * 180 / pi
        y = (torch.squeeze(y) * (data_stats['gaze']['max'] - data_stats['gaze']['min']) + data_stats['gaze'][
            'min']) * 180 / pi
        error = abs((out - y))
        loss_3d = loss_3d * 180 / pi
        total_test_loss += loss_3d
        result.append(
            [out[0].item(), out[1].item(), y[0].item(), y[1].item(), loss_3d, error[0].item(), error[1].item()])
        count += 1
    # Obtain a Dataframe having all required parameters related to the test
    res_df = pd.DataFrame(result, columns=['yaw_p', 'pitch_p', 'yaw_t', 'pitch_t', 'loss_3d', 'error_y', 'error_p'])
    mean_test_loss = float(total_test_loss) / count
    std_test_loss = res_df['loss_3d'].std()
    print(('Completed test with test loss = ' + str(mean_test_loss) + ' and std = ' + str(std_test_loss)))
    logging.info('Completed test with test loss = ' + str(mean_test_loss) + ' and std = ' + str(std_test_loss))
    if save_preds and train_dataset == dataset:
        filename = os.path.split(os.path.split(weight_path)[0])[-1] + '_' + \
                   os.path.splitext(os.path.split(weight_path)[-1])[0]
        res_df.to_csv(os.path.join(test_path, filename + '.csv'))
    return mean_test_loss, std_test_loss
