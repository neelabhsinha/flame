import logging
import os
import pickle
from math import pi

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, BatchSampler

from config import dataset_paths, loggers_loc, test_path, project_path
from datasets.data import Data, Sampler
from losses.angular_loss import AngularGazeLoss
from models.additive_fusion import AdditiveFusionNet
from models.aggregation_only import ConcatenatedFusionNet
from models.baseline import BaselineNetwork
from models.mmtm_fusion import MMTMFusionNet
from utils.train import forward_propagation


def test_model(weight_path, network_name, train_dataset, test_data, resolution=120, split_nature='cross-person', save_preds=True):
    """
    executes test of a model on a given dataset
    :param weight_path: path to the checkpoint containing weights and other data
    :param network_name: name of the network
    :param train_dataset: name of the train dataset
    :param test_data: name of the test dataset
    :param resolution: resolution of the input image and heatmap used for training and testing
    :param split_nature: nature of split - cross-person
    :param save_preds: boolean to specify whether to save predictions or heatmap or not
    :return: void
    """
    dataset = test_data
    frame_window = 1
    logging.basicConfig(
        filename=loggers_loc + '/testing_' + train_dataset + '_' + network_name + '_' + split_nature + '.log',
        format='%(asctime)s %(levelname)s : %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p',
        level=logging.INFO)
    logging.info('Starting testing of ' + network_name + ' on ' + dataset + ' for ' + split_nature + ' split')

    # Load Data (Add the dataset path in config.py if adding new)
    try:
        path = dataset_paths[dataset]
    except KeyError:
        logging.error('Path to dataset ' + dataset + ' not defined. Please define the same in config.py file')
        sys.exit()
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
    if save_preds:
        filename = train_dataset + '_' + test_data + '_' + network_name
        res_df.to_csv(os.path.join(test_path, filename + '.csv'))
    return mean_test_loss, std_test_loss
