import logging
import os
import pickle
import sys
from datetime import datetime
from math import pi
from subprocess import call

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import BatchSampler
from torch.utils.data import DataLoader

from config import dataset_paths, weights_path, loggers_loc, project_path
from datasets.data import Data, Sampler
from losses.angular_loss import AngularGazeLoss
from losses.vector_loss import VectorDifferenceLoss
from models.additive_fusion import AdditiveFusionNet
from models.aggregation_only import ConcatenatedFusionNet
from models.baseline import BaselineNetwork
from models.mmtm_fusion import MMTMFusionNet


def print_trainable_parameters(model):
    '''
    prints number of trainable parameters of a pytorch model
    :param model: pytorch model
    :return: void
    '''
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
    print('Trainable Parameters: %.3fM' % parameters)


def print_cuda_details():
    '''
    prints cuda details
    :return: void
    '''
    print('__Python VERSION:', sys.version)
    print('__pyTorch VERSION:', torch.__version__)
    print('__CUDA VERSION')

    # call(["nvcc", "--version"]) does not work
    print('__CUDNN VERSION:', torch.backends.cudnn.version())
    print('__Number CUDA Devices:', torch.cuda.device_count())
    print('__Devices')
    call(["nvidia-smi", "--format=csv", "--query-gpu=index,name,driver_version,memory.total,memory.used,memory.free"])
    print('Active CUDA Device: GPU', torch.cuda.current_device())

    print('Available devices ', torch.cuda.device_count())
    print('Current cuda device ', torch.cuda.current_device())


def train_network(network_name, dataset, epochs, frame_window, batch_size, split_nature='cross-person',
                  two_phase=False, checkpoint=None):
    """
    function to train the neural network with given parameters
    :param network_name: name of the network as defined
    :param dataset: name of the dataset as defined
    :param epochs: number of epochs to train
    :param frame_window: by default considered as 1 as we are taking one frame at a time
    :param batch_size: batch size
    :param split_nature: cross-person (person-independent)/random (only for EYEDIAP)
    :param two_phase: whether to use a two-phase training or not (CURRENTLY NOT BEING USED)
    :param checkpoint: pth file of the checkpoint to load the model with, if any
    :return: void
    """

    resolution = 120
    # Define Logger
    logging.basicConfig(
        filename=loggers_loc + '/training_' + dataset + '_' + network_name + '_' + split_nature + '.log',
        format='%(asctime)s %(levelname)s : %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
    logging.info('Starting training of ' + network_name + ' on ' + dataset + ' for ' + split_nature + ' split')
    learning_rate = 0.0001

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

    # Load Data (Add the dataset path in config.py if adding new)
    try:
        path = dataset_paths[dataset]
    except KeyError:
        logging.error('Path to dataset ' + dataset + ' not defined. Please define the same in config.py file')
        sys.exit()

    # Define Sampler and dataset
    train_batch_sampler = BatchSampler(Sampler(dataset, path, batch_size, frame_window, 'train', split_nature),
                                       batch_size, drop_last=True)
    cv_batch_sampler = BatchSampler(Sampler(dataset, path, 3, frame_window, 'cv', split_nature),
                                    16, drop_last=True)
    train_data_set = Data(dataset, path, frame_window, 'train', resolution=resolution, split_nature=split_nature,
                          crop='eye')
    cv_data_set = Data(dataset, path, frame_window, 'cv', resolution=resolution, split_nature=split_nature, crop='eye')
    logging.info('Samplers and Dataset created for ' + dataset)

    # Change to GPU if available
    if torch.cuda.is_available():
        model = model.cuda()
    # Make the model run on Multiple GPUs if available
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model, device_ids=range(0, torch.cuda.device_count()))

    logging.info(network_name + ' model defined for training')
    logging.info('Two Phase Training - ' + str(two_phase))

    # Define Data Loaders
    train_loader = DataLoader(train_data_set, batch_sampler=train_batch_sampler)
    cv_loader = DataLoader(cv_data_set, batch_sampler=cv_batch_sampler)
    logging.info('DataLoaders initialized')

    # Define Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    logging.info('Optimizer initialized')

    # Define Scheduler
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 30, 40], gamma=0.5)
    logging.info('Scheduler initialized')

    # Load Checkpoint (if any)
    if checkpoint is not None:
        logging.info('Previous checkpoint found. Loading...')
        if torch.cuda.is_available():
            cp = torch.load(checkpoint, map_location=torch.device('cuda'))
        else:
            cp = torch.load(checkpoint, map_location=torch.device('cpu'))
        logging.info('Checkpoint details -')
        logging.info('Scheduler state - ' + str(cp['scheduler_state_dict']))
        print('Scheduler state - ' + str(cp['scheduler_state_dict']))
        model.load_state_dict(cp['model_state_dict'])
        optimizer.load_state_dict(cp['optimizer_state_dict'])
        scheduler.load_state_dict(cp['scheduler_state_dict'])
        epoch_start = cp['epoch'] + 1
        checkpoint_dir = os.path.split(checkpoint)[0]
    else:
        logging.info('No previous checkpoint available. Training initialized from scratch.')
        epoch_start = 0
        now = datetime.now()
        dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")
        foldername = network_name + '_' + dataset + '_' + dt_string
        checkpoint_dir = os.path.join(weights_path, foldername)

    # Loading Max Min metadata for normalization
    with open(os.path.join(project_path, 'metadata', 'data_statistics', 'data_vals_max_min_' + dataset + '.pkl'),
              'rb') as fm:
        data_stats = pickle.load(fm)
        fm.close()

    # Define Loss Functions
    eval_criterion = AngularGazeLoss(data_stats)
    criterion = VectorDifferenceLoss(data_stats)
    if torch.cuda.is_available():
        criterion = criterion.cuda()
        eval_criterion = eval_criterion.cuda()

    adj = None

    # Training Loop
    for epoch in range(epoch_start, epochs):
        count = 0
        total_train_loss = 0
        model.train()
        # Freeze Backbone weights for two-phase training
        if epoch >= 179 and two_phase:  # Two-phase training is not a paer of actual methodogies produced as they give sub-optimal results
            for parameters in model.backbone.parameters():
                parameters.requires_grad = False
        for step, data in enumerate(train_loader):
            # Forward Propagation
            loss, out, y = forward_propagation(model, data, adj, 'train', data_stats, network_name, criterion)
            total_train_loss += loss.item()
            count += 1
            # Backward Propagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        mean_train_loss = float(total_train_loss) / count
        count = 0
        total_cv_loss = 0
        model.eval()
        for step, data in enumerate(cv_loader):
            loss, out, y = forward_propagation(model, data, adj, 'eval', data_stats, network_name, eval_criterion)
            total_cv_loss += (loss * 180 / pi)
            count += 1
        mean_cv_loss = float(total_cv_loss) / count
        # Create log
        logging.info('Completed epoch ' + str(epoch) + ':' + ' training loss = ' + str(
            mean_train_loss) + ' validation loss = ' + str(mean_cv_loss))
        # Save Checkpoint
        if not os.path.exists(checkpoint_dir):
            os.mkdir(checkpoint_dir)
        filepath = os.path.join(checkpoint_dir, 'checkpoint_' + str(epoch + 1) + '.pth')
        # Save checkpoint
        torch.save(
            {'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(),
             'scheduler_state_dict': scheduler.state_dict(), 'loss_train': mean_train_loss, 'loss_cv': mean_cv_loss},
            filepath)
        scheduler.step()


def forward_propagation(model, data, adj, task, data_stats, network_name, criterion=None):
    """
    Forward propagation method that executes forward pass of the given data and model
    :param model: pytorch model
    :param data: data batch
    :param adj: adjacency matrix (if GCN is used)
    :param task: train/eval
    :param data_stats: metadata for the dataset like maximum, minimum values for normalization
    :param network_name: name of the network as defined
    :param criterion: loss function
    :return: loss tensor, prediction tensor and ground truth tensor for that particular batch
    """
    rgb = data["frames"].float()
    fl = (data["fl"].float() - data_stats['fl']['min']) / (data_stats['fl']['max'] - data_stats['fl']['min'])
    heatmap = data["heatmaps"].float()
    hp = (data["hp"].float() - data_stats['hp']['min']) / (data_stats['hp']['max'] - data_stats['hp']['min'])
    y = (data["gaze"].float() - data_stats['gaze']['min']) / (data_stats['gaze']['max'] - data_stats['gaze']['min'])
    # Convert to GPU if CUDA is available
    if torch.cuda.is_available():
        rgb = rgb.cuda(non_blocking=True)
        fl = fl.cuda(non_blocking=True)
        heatmap = heatmap.cuda(non_blocking=True)
        hp = hp.cuda(non_blocking=True)
        y = y.cuda(non_blocking=True)
        adj = adj.cuda(non_blocking=True) if adj is not None else None
    # Forward Propagation
    if model.__class__.__name__ == 'DenseFusionNet' or model.__class__.__name__ == 'DenseBaselineNet':
        out = model(rgb, fl, hp, adj)
    elif network_name == 'heatmap-baseline':
        out = model(heatmap, rgb, hp)
    elif network_name == 'early-fusion':
        out = model(torch.cat((rgb, heatmap), 1), heatmap, hp)
    else:
        out = model(rgb, heatmap, hp)
    # Loss Calculation
    loss = criterion(y, out)
    if task == 'eval':  # for optimization of memory
        loss = loss.item()
    return loss, out, y
