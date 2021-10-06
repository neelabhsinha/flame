import os
from config import eyediap_processed_data, project_path, columbiagaze_processed_data, mpiigaze_processed_data
from random import shuffle
import pickle


def split_data(dataset, split_nature, split_fraction=(0.8,0.1,0.1)):
    """
    split data into train-test-cv
    :param dataset: dataset name
    :param split_nature: nature of split (random/cross-person)
    :param split_fraction: fraction os split in (train,test,cv)
    :return: void (saves result into pickle)
    """
    split = {}
    split['train'] = []
    split['test'] = []
    split['cv'] = []
    if dataset == 'eyediap':
        srcpath = os.path.join(eyediap_processed_data, 'images')
        videos = os.listdir(srcpath)
    elif dataset == 'columbiagaze':
        srcpath = os.path.join(columbiagaze_processed_data, 'images')
        videos = os.listdir(srcpath)
    if type == 'random' or dataset != 'eyediap':
        shuffle(videos)
        count = 0
        for vid in videos:
            if count < split_fraction[0]*len(videos):
                split['train'].append(vid)
            elif split_fraction[0]*len(videos) < count < (split_fraction[0] + split_fraction[2])*len(videos):
                split['cv'].append(vid)
            else:
                split['test'].append(vid)
            count += 1
    elif type == 'cross-person' and dataset == 'eyediap':
        person_videos = {}
        for vids in videos:
            person = vids.split('_')[0]
            if person not in person_videos:
                person_videos[person] = []
            person_videos[person].append(vids)
        people = list(person_videos.keys())
        shuffle(people) #randomisation of people
        count = 0
        for p in people:
            videos = person_videos[p]
            if count < split_fraction[0] * len(people):
                for vid in videos:
                    split['train'].append(vid)
            elif split_fraction[0] * len(people) < count < (split_fraction[0] + split_fraction[2]) * len(people):
                for vid in videos:
                    split['cv'].append(vid)
            else:
                for vid in videos:
                    split['test'].append(vid)
            count += 1

    with open(os.path.join(project_path, 'metadata','splits','data_split_' + dataset + '_' + split_nature + '.pkl'), 'wb') as f:
        pickle.dump(split, f)
