import os 
import argparse

from dataloaders.MMWHS2017DataLoader import MMWHS2017Dataset 
from dataloaders.BraTS2020DataLoader import BraTS2020Dataset 
from dataloaders.IXIDataLoader import IXIDataset
from dataloaders.ACRINDataLoader import ACRINDataset
from dataloaders.ProstateMRIDataLoader import ProstateDataset
from dataloaders.RESECTDataLoader import RESECTDataset
import dataloaders.Transforms as RT 

from torch.utils.data import DataLoader 


# volume's shape in different dataset 
DATA_SHAPE_MAPPING = {
    'brats2020': (155, 240, 240), # -> (128, 192, 192)
    'mmwhs2017': (96, 96, 96),
    'ixi'      : (192, 256, 256), # -> (128, 160, 160)
    'acrin'    : (160, 256, 256), # -> (128, 192, 192)
    'prostate' : (160, 256, 256), # -> (128, 192, 192)
    'resect'   : (192, 256, 256), # -> (128, 160, 160)
}

# dataset sub dir name to find data 
DIR_NAME_MAPPING = {
    'brats2020': 'BraTS2020_Brain_MRI_MM',
    'mmwhs2017': 'MMWHS2017_Heart_CT_MR',  
    'ixi'      : 'IXI-T1-T2-PD',
    'acrin'    : 'ACRIN-HNSCC-FDG-PET-CT',
    'prostate' : 'Prostate-Fused-MRI',
    'resect'   : 'RESECT',
}

# instantize dataset 
DATASET_MAPPING = {
    'brats2020': BraTS2020Dataset,
    'mmwhs2017': MMWHS2017Dataset,
    'ixi'      : IXIDataset,
    'acrin'    : ACRINDataset,
    'prostate' : ProstateDataset,
    'resect'   : RESECTDataset,
} 


def get_transforms(args): 
    """ Return an unified data transforms for all datasets. """ 

    resize_trans = None 
    if args.is_resize: 
        target_shape = tuple([int(x) for x in args.target_shape.split(',')]) 
        resize_trans = RT.TargetResize(target_shape) 

    train_transforms = RT.Compose([
        RT.AdjustNumpyType(),
        RT.RandomFlip(), 
        RT.Normalize(), 
        RT.AdjustChannels(), 
        resize_trans, 
        RT.ToTensor() 
    ]) 

    valid_test_transforms = RT.Compose([
        RT.AdjustNumpyType(),
        RT.Normalize(), 
        RT.AdjustChannels(), 
        resize_trans, 
        RT.ToTensor() 
    ]) 

    return train_transforms, valid_test_transforms 

   
# directly use this function to get your dataloader 
def get_dataloader(args): 
    args.is_resize = True
    if args.target_shape is None:
        if args.which_set == 'brats2020':
            args.target_shape = '128,192,192'
        elif args.which_set == 'ixi':
            args.target_shape = '128,160,160'
        elif args.which_set == 'acrin':
            args.target_shape = '128,192,192'
        elif args.which_set == 'prostate':
            args.target_shape = '128,192,192'
        elif args.which_set == 'resect':
            args.target_shape = '128,160,160'
        
    train_trans, valid_test_trans = get_transforms(args) 

    args.root_dir = os.path.join(args.dataroot, DIR_NAME_MAPPING[args.which_set]) 

    train_dataset = DATASET_MAPPING[args.which_set](args.root_dir, True, args.mode1, args.mode2, train_trans) 
    valid_test_dataset = DATASET_MAPPING[args.which_set](args.root_dir, False, args.mode1, args.mode2, valid_test_trans) 

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True) 
    valid_test_dataloader = DataLoader(valid_test_dataset, batch_size=1, shuffle=False) 

    return train_dataloader, valid_test_dataloader

