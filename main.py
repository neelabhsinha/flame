import os
import pickle
import argparse

from utils.preprocess import get_mean_and_std, get_and_save_heatmap
from utils.data_split import split_data
from utils.train import train_network
from config import epochs, frame_window, batch_size, project_path
from utils.test import test_model


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=None, help='specify dataset under operation')
    parser.add_argument('--get_heatmap', default=False, action='store_true', help='Generate heatmaps from Facial Landmarks')
    parser.add_argument("--split_data", default=False, action='store_true',
                        help='split a given split of the data into train-test-cv')
    parser.add_argument("--split_nature", default='cross-person', help='nature of split into train-test-cv')
    parser.add_argument("--train", default=None, help='to train a given model with name of the model')
    parser.add_argument("--test", default=None, help='to train a given model with complete path of the weights')
    parser.add_argument("--test_data", default=None, help='to train a given model with complete path of the weights')
    parser.add_argument('--load_checkpoint', default=None, help='load a given checkpoint with path')
    parser.add_argument('--get_data_stats', default=False, action='store_true', help='get mean, std of data with path of dataset and split nature')
    parser.add_argument('--two-phase-training', default=False, action='store_true', help='enable two-phase training for FL modality')
    args = parser.parse_args()
    return args


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    args = get_args()
    data = args.dataset
    split = args.split_data
    split_type = args.split_nature
    training_network = args.train
    test_network = args.test
    stats_calc = args.get_data_stats
    checkpoint_path = None if args.load_checkpoint == 'no_checkpoint' else args.load_checkpoint
    if args.get_heatmap:
        get_and_save_heatmap(data)
    if args.split_data:
        if data == 'eyediap' and split_type == 'cross-person':
            split_data(data, split_type, (0.6, 0.2, 0.2))
        else:
            split_data(data, split_type)
        # get_mean_and_std(data, split_type)
    if stats_calc:
        get_mean_and_std(data, split_type)
    if training_network is not None:
        train_network(training_network, data, epochs, frame_window, batch_size, split_type, args.two_phase_training, checkpoint_path)
    if test_network is not None:
        test_model(frame_window, test_network, split_type, args.test_data)
