# FLAME
Original Pytorch Implementation of FLAME: Facial Landmark Heatmap Activated Multimodal Gaze Estimation, accepted at the 17th IEEE Internation Conference on Advanced Video and Signal-based Surveillance, AVSS 2021, Virtual, November 16-19, 2021.

arXiv Preprint: [arXiv](https://google.com)

## Dependencies used
- python v3.6.13
- cuda v10.2
- numpy v1.19.2
- pandas v1.1.3
- opencv-python-headless v4.5.1.48 (or use equivalent opencv)
- torch v1.8.0
- torchvision v0.9.0
- scipy v1.5.2
- matplotlib v3.3.4

## Prepraring the dataset

Prepare a directory structure as the following and add the root in the dataset_paths dictionary in [config.py](config.py).
```
dataset_root
│
└───images
│   │───subject01
│   │   │   1.npy
│   |   │   2.npy
│   |   │   ...
│   │
│   └───subject02
│   |   │   1.npy
│   |   │   2.npy
│   |   │   ...
│   |....
│   
└───heatmaps
│   │───subject01
│   │   │   1.npy
│   |   │   2.npy
│   |   │   ...
│   │
│   └───subject02
│   |   │   1.npy
│   |   │   2.npy
│   |   │   ...
│   |....
|
└───facial_landmarks_2d
│   │───subject01.pkl
│   |───subject02.pkl
|   |....
│   
|
└───head_pose
│   │───subject01.pkl
│   |───subject02.pkl
|   |....
│   
└───gaze_target
│   │───subject01.pkl
│   |───subject02.pkl
|   |....
│   
```
- images - RGB images as numpy arrays with (height,width,channels). Stored as one folder for each subject, and images inside those (as npy arrays).
- heatmaps - 2D landmark heatmaps with (height, width, channels). Stored as one folder for each subject, and images inside those (as npy arrays).
- facial_landmarks_2d - 2D facial landmarks in the form of numpy arrays saved as pickle (index i of a file contains data corresponding to (i+1)th image and heatmap file inside the given subject folder as that of the pickle file).
- head_pose - Head pose angles in the form of numpy arrays of the type [pitch,yaw,roll] angle numpy arrays (index i of a file contains data corresponding to (i+1)th image and heatmap file inside the given subject folder as that of the pickle file). If not available, headpose can be used from the output of [Openface 2.0](https://github.com/TadasBaltrusaitis/OpenFace).
- gaze_angles - Gaze angles in the form of numpy arrays of the type [yaw, pitch] angle numpy arrays (index i of a file contains data corresponding to (i+1)th image and heatmap file inside the given subject folder as that of the pickle file).

Steps - 
1. Extract Face crops from [RetinaFace](https://github.com/serengil/retinaface) and zero-pad them to nearest 4:5 ratio
2. Crop them to 384 * 450 pixels
3. Run [Openface 2.0](https://github.com/TadasBaltrusaitis/OpenFace) on these images
4. Collect the 2D facial landmarks from it in the above directory structure as per the given instruction above
5. Collect the images, head pose, and gaze targets in the above directory structure as per given instructions. To generate head_pose angles from rotation matrix, use get_head_pose_angles from [utils/preprocess.py](utils/preprocess.py)
6. Add the root dataset directory to dataset_paths by dataset_name:dataset_path in [config.py](config.py) (Use this **dataset_name** everywhere in the code for all dataset name related parameters in the code)
7. Generate heatmaps from the 2D landmarks after completing step 1-6. You can use the function get_and_save_heatmap given in [utils/preprocess.py](utils/preprocess.py) with dataset_name as parameter. Use the following command -

```
$ python3 main.py --get_heatmap --dataset <dataset_name>
```

9. It should create heatmaps directory and save the heatmaps there.

Notes
- Maintain 1...n continuous numbers for images, heatmaps and save other data in pickle at corresponding 0-(n-1) indices
- Take care of the file formats

## Other Configurations required

Please do the following before running the code
1. Please add all the dependencies in your environment which support the given version
2. In config.py file, change/add all the dataset paths, and other parameters as defined

## Code Explanation

The functions in the code have doctext which describe the purpose of the function and their parameters, return type

Each model as described in the paper identified with a unique key in this code which we shall address by **model_key** in this readme. The keys to those models are defined below -

|Model Name | Model key (model_key) | Model definition |
|-----------|-----------------------|------------------|
|FLAME|mmtm-fusion|models/mmtm_fusion.py|
|F-AO|concatenated-fusion|models/aggregation_only.py|
|F-AF|additive-fusion|models/additive_fusion.py|
|F-B|baseline|models/baseline.py|

We shall explain how to use this model key when we cover how to run the code in the below sections.

**Pretrained Weights** -

| Model Name | Model Key | EYEDIAP | Columbiagaze |
|------------|-----------|---------|--------------|
| FLAME | mmtm-fusion | [Checkpoint]() | [Checkpoint]()|
| F-AO | concatenated-fusion | [Checkpoint]() | [Checkpoint]()|
| F-AF | additive-fusion | [Checkpoint]() | [Checkpoint]()|
| RGB Baseline | baseline | [Checkpoint]() | [Checkpoint]()|

The structure of these checkpoints is in the form of dictionary with following schema -

```
{
  'epoch': epoch number, 
  'model_state_dict': model_weights, 
  'optimizer_state_dict': optimizer state,
  'scheduler_state_dict': scheduler_state,
  'loss_train': mean_training_loss, 
  'loss_cv': mean_cv_loss
}
```

## Optional Configurations

Few other metadata that is required but is already given along with this repository for our experiments are described below. You may run it on your own but it's not compulsory.

1. **Generating Split** - Decide which folders will be in train, test, and val splits. Can be done using the following script (our split is available in [metadata/splits](metadata/splits) directory) -

```
$ python3 main.py --split_data --split_nature cross-person --data <dataset_name>
```

Function is available at [utils/data_split.py](utils/data_split.py) for viewing the schema of the file

2. **Getting maximum and minimum values of each input and output** - Used for normalization purposes and is extracted only for the training data. For our split and data, the parameters are given in [metadata/data_statistics](metadata/data_statistics) in the form of a dictionary stored as pickle. Use the following command to extract these parameters -

```
$ python3 main.py --get_data_stats --dataset <dataset_name>
```
Function is available at [utils/preprocess.py](utils/preprocess.py) by the name get_mean_and_std

## Evaluations/Testing

1. Set up the project using above steps
2. Download the weights from the specified locations
3. Execute the following command -

```
$ python3 main.py --test <model_key> --dataset <dataset_name of training dataset> --test_data <dataset_name of testing dataset> --load_checkpoint <complete path of the checkpoint on which model is to be tested>
```

A csv file will be stored at the test_path location as specified in config.py by the name '<train_dataset_name>_<test_dataset_name>_<model_key>.csv' having the following schema for all provided images in order -
```
index, yaw_p, pitch_p, yaw_t, pitch_t, loss_3d, error_y, error_p
```

Note -
- To generate predictions on the customized pipeline, you can create an input pipeline on your own and use the function forward_propagation inside [utils/train.py](utils/train.py) and provide the inputs to the same. It will return you the values in order of a tuple ((predicted_yaw, predicted_pitch),(true_yaw, true_pitch), error) of type (tensor, tensor, float).

## Training

1. Set up the project using above steps
2. Execute the following command -

```
$ python3 main.py --train <model_key> --dataset <dataset_name>
```

To change training hyperparameters, change variables in [config.py](config.py) file

Training from a checkpoint -
```
$ python3 main.py --train <model_key< --dataset <dataset_name> --load_checkpoint <complete path of checkpoint file>
```
## Citation

If you found our work helpful in your use case, please cite the following paper -



  




