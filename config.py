# --- Overall Project and Dataset Paths --- #
project_path = '/home/nsinha/projects/flamge'  # path to the code of the project
dataset_paths = {
    'eyediap': '/data/stars/user/nsinha/flamge/data/eyediap/final',  # path to the final data location of EYEDIAP
    'columbiagaze': '/data/stars/user/nsinha/flamge/data/columbiagaze/final'  # path to the final data location of ColumbiaGaze
}
# Add other datasets in the dictionary

# --- Training and Testing --- #
weights_path = '/data/stars/user/nsinha/flamge/weights'  # path to save the weights and training checkpoints
test_path = '/data/stars/user/nsinha/flamge/test_results'  # path to save test results
loggers_loc = '/data/stars/user/nsinha/flamge/loggers'  # path to save log files

# Note - Make sure the above directories exist (even if empty)


# Training Hyperparameters
epochs = 201
batch_size = 8
frame_window = 1  # keep it 1 by default

# --- Fine tuning --- #
finetune_weights_path = '/data/stars/user/nsinha/flamge/finetune_weights'
finetune_epochs = 56
