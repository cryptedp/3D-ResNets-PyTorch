import torch
import torch.utils.data as data
from PIL import Image
import os
import math
import functools
import json
from tensorpack.dataflow import *
import numpy as np
import copy

from utils import load_value_file


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def accimage_loader(path):
    try:
        import accimage
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def get_default_image_loader():
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader
    else:
        return pil_loader


def video_loader(video_dir_path, frame_indices, image_loader):
    video = []
    for i in frame_indices:
        image_path = os.path.join(video_dir_path, 'image_{:05d}.jpg'.format(i))
        if os.path.exists(image_path):
            video.append(image_loader(image_path))
        else:
            return video

    return video


def get_default_video_loader():
    image_loader = get_default_image_loader()
    return functools.partial(video_loader, image_loader=image_loader)


def load_annotation_data(data_file_path):
    with open(data_file_path, 'r') as data_file:
        return json.load(data_file)


def get_video_names_and_annotations(data, subset):
    video_names = []
    annotations = []
    for i in range(len(data)):
        current_data = data[i]

        this_subset = current_data.get("source487")
        if this_subset == subset:
            if subset == 'testing':
                video_names.append('test/{}'.format(current_data['id']))
            else:
                labels = current_data['label487']
                # video_names.append("v_"+current_data['id'])
                annotations.append(labels)
    video_names = [name for name in os.listdir('/data/sammer/sports_1m/jpg')]

    return video_names, annotations


def make_dataset(root_path, annotation_path, subset,
                 n_samples_for_each_video, sample_duration):
    
    data = load_annotation_data('/home/sammer/sports1m_train.json')
    video_names, annotations = get_video_names_and_annotations(data, subset)

    # return random labels
    
    idx_to_class = randomize_labels(annotations)
    dataset = []
    # n_frames_file_path = os.path.join(video_path,'n_frames')
    # print(n_frames_file_path)
    for i in range(len(video_names)):
        if i % 1000 == 0:
            print('dataset loading [{}/{}]'.format(i, len(video_names)))

        video_path = os.path.join('data/sammer/sports_1m/jpg/', video_names[i])
        # print(video_path)
        if not os.path.exists(video_path):
            continue

        n_frames_file_path = os.path.join(video_path, 'n_frames')
        # print(n_frames_file_path)
        n_frames = int(load_value_file(n_frames_file_path))
        if n_frames <= 0:
            continue

        begin_t = 1
        end_t = n_frames
        sample = {
            'video': video_path,
            'segment': [begin_t, end_t],
            'n_frames': n_frames,
            'video_id': video_names[i]
        }
        if len(annotations) != 0:
            sample['label'] = annotations[i]
        else:
            sample['label'] = -1

        if n_samples_for_each_video == 1:
            sample['frame_indices'] = list(range(1, n_frames + 1))
            dataset.append(sample)
        else:
            if n_samples_for_each_video > 1:
                step = max(1, math.ceil((n_frames - 1 - sample_duration) / (n_samples_for_each_video - 1)))
            else:
                step = sample_duration
            for j in range(1, n_frames, step):
                sample_j = copy.deepcopy(sample)
                sample_j['frame_indices'] = list(range(j, min(n_frames + 1,
                                                              j + sample_duration)))
                dataset.append(sample_j)

    # print(dataset)
    return dataset, idx_to_class

    # randomize labels


def randomize_labels(class_names):
    labels = np.array(class_names)
    np.random.seed(12345)
    mask = np.random.rand(len(labels)) <= 1
    rnd_labels = np.random.choice(487, mask.sum())
    labels[mask] = rnd_labels
    # we need to explicitly cast the labels from npy.int64 to
    # builtin int type, otherwise pytorch will fail...
    labels = [int(x) for x in labels]
    return labels


class sports1m(data.Dataset):
    """
    Args:
        root (string): Root direictory path.
        spatial_transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        temporal_transform (callable, optional): A function/transform that  takes in a list of frame indices
            and returns a transformed version
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an video given its path and frame indices.
     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(self, root_path, annotation_path, subset, n_samples_for_each_video=1,
                 spatial_transform=None, temporal_transform=None, target_transform=None,
                 sample_duration=16, get_loader=get_default_video_loader):
        print(root_path)
        self.data, self.class_names = make_dataset(root_path, annotation_path, subset,
                                                   n_samples_for_each_video, sample_duration)

        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform
        self.target_transform = target_transform
        self.loader = get_loader()

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:p
            tuple: (image, target) where target is class_index of the target class.
        """
        path = self.data[index]['video']

        frame_indices = self.data[index]['frame_indices']
        if self.temporal_transform is not None:
            frame_indices = self.temporal_transform(frame_indices)
        clip = self.loader(path, frame_indices)
        if self.spatial_transform is not None:
            self.spatial_transform.randomize_parameters()
            clip = [self.spatial_transform(img) for img in clip]
        clip = torch.stack(clip, 0).permute(1, 0, 2, 3)

        target = self.class_names
        
        #if self.target_transform is not None:
            #target = self.target_transform(target)

        return clip, target

    def __len__(self):
        return len(self.data)




