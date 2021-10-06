# --- Overall Project Path --- #
project_path = '/home/nsinha/projects/flamge'  # path to the code of the project
eyediap_processed_data = '/data/stars/user/nsinha/flamge/data/eyediap/final'  # path to the final data location of EYEDIAP
columbiagaze_processed_data = '/data/stars/user/nsinha/flamge/data/columbiagaze/final'  # path to the final data location of ColumbiaGaze
mpiigaze_processed_data = '/data/stars/user/nsinha/flamge/data/mpiigaze/final'  # path to the final data location of MPIIGaze
# Add other datasets

# --- Training --- #
weights_path = '/data/stars/user/nsinha/flamge/weights'  # path to save the weights and training checkpoint
test_path = '/data/stars/user/nsinha/flamge/test_results'  # path to save test results
loggers_loc = '/data/stars/user/nsinha/flamge/loggers'  # path to save log files
train_report = '/data/stars/user/nsinha/flamge/report/training'  # path to save training report
test_report = '/data/stars/user/nsinha/flamge/report/test'  # path to save test reports

# Training Hyperparameters
epochs = 201
batch_size = 8
frame_window = 1  # keep it 1 by default

# --- Fine tuning --- #
finetune_weights_path = '/data/stars/user/nsinha/flamge/finetune_weights'
finetune_epochs = 56
