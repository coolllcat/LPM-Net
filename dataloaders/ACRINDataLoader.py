import os 
import torch 
import numpy as np 
import SimpleITK as sitk 

from torch.utils.data import Dataset, DataLoader 
from typing import List, Tuple, Dict 


def sort_acrin_dirs(names: List[str]) -> List[str]: 
    indices = sorted(
        range(len(names)), 
        key=lambda index: int(names[index].split('_')[1]) 
    ) 
    sorted_names = [names[index] for index in indices] 
    return sorted_names


def remove_intro_files(sample_names: List[str]): 
    name_buffer = [] 
    for sample_name in sample_names: 
        if sample_name.find('.') == -1: 
            name_buffer.append(sample_name) 
    return name_buffer 


def parse_indices(txt_path: str) -> List[int]: 
    res = [] 

    with open(txt_path, 'r') as f: 
        for line in f.readlines(): 
            res.append(int(line)) 

    return res  


class ACRINDataset(Dataset): 
    def __init__(self, root_dir: str, is_train: bool, mode1: str = 'ct', mode2: str = 'pet', transforms=None) -> None:
        super().__init__() 

        _supported_domains = ['ct', 'pet']
        assert mode1 in _supported_domains and mode2 in _supported_domains, \
        "mode1 and mode2 should be in {}, but got {}, {}".format(
            _supported_domains, mode1, mode2)
        
        self.root_dir = root_dir
        self.is_train = is_train
        self.mode1 = mode1 
        self.mode2 = mode2 

        sample_names = os.listdir(root_dir) 
        sample_names = remove_intro_files(sample_names) 
        sample_names = sort_acrin_dirs(sample_names) 

        train_indices_path = os.path.join(root_dir, 'IndicesTrain.txt') 
        self.train_indices = parse_indices(train_indices_path) 
        valid_test_indices_path = os.path.join(root_dir, 'IndicesTest.txt') 
        self.valid_test_indices = parse_indices(valid_test_indices_path) 

        self.train_sample_dirs = [os.path.join(root_dir, sample_names[i]) for i in self.train_indices] 
        self.valid_test_sample_dirs = [os.path.join(root_dir, sample_names[i]) for i in self.valid_test_indices] 

        print("MMWHS2017 data split: train: {}, valid and test: {}".format(
            len(self.train_indices), len(self.valid_test_indices)
        )) 

        self.transforms = transforms 

    @staticmethod 
    def _load_image_and_label(sample_dir: str, mode1: str, mode2: str): 

        mode1_image_path = os.path.join(sample_dir, "{}.nii.gz".format(mode1)) 
        mode1_image = sitk.GetArrayFromImage(sitk.ReadImage(mode1_image_path))

        mode2_image_path = os.path.join(sample_dir, "{}.nii.gz".format(mode2)) 
        mode2_image = sitk.GetArrayFromImage(sitk.ReadImage(mode2_image_path))
         
        return mode1_image, mode2_image 
    
    def __getitem__(self, index): 
        if self.is_train: 
            mode1_image, mode2_image = self._load_image_and_label(self.train_sample_dirs[index], self.mode1, self.mode2)
        else: 
            mode1_image, mode2_image = self._load_image_and_label(self.valid_test_sample_dirs[index], self.mode1, self.mode2)
        
        # transform here 
        if self.transforms != None: 
            mode1_image, mode2_image = self.transforms(mode1_image, mode2_image)
        
        return mode1_image, mode2_image 
    
    def __len__(self): 
        if self.is_train: 
            return len(self.train_sample_dirs) 
        else: 
            return len(self.valid_test_sample_dirs) 
        
    def __str__(self) -> str:
        return "ACRINDataset" 


if __name__ == '__main__':
    dataset = ACRINDataset('/data/postgraduate/wmw/FusionDataset/ACRIN-HNSCC-FDG-PET-CT', True)

    mode1_image, mode2_image = dataset[0]

    print(dataset.__len__())
    print(mode1_image.shape)
    print(mode2_image.shape)