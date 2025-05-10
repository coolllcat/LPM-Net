import math 
import numpy as np 
import torch 
import torch.nn.functional as F 

import skimage.transform as T 

from typing import List, Tuple, Dict 


class AdjustNumpyType(object): 
    def __init__(self): 
        super().__init__() 

    def __call__(self, mode1_image, mode2_image): 
        mode1_image = np.array(mode1_image, dtype='float32') 
        mode2_image = np.array(mode2_image, dtype='float32') 
        
        return mode1_image, mode2_image 
    
    def __str__(self) -> str:
        return "Adjust image type to float32 label type to int32"


class RandomRotation(object): 
    def __init__(self, dims: List[Tuple[int, int]] = [(0, 1), (1, 2), (2, 0)]): 
        self.dims = dims 

    def __call__(self, mode1_image, mode2_image): 
        for dim in self.dims: 
            rand_k = np.random.randint(0, 4) 
            mode1_image = np.rot90(mode1_image, k=rand_k, axes=dim) 
            mode2_image = np.rot90(mode2_image, k=rand_k, axes=dim) 
            
        return mode1_image, mode2_image

    def __str__(self) -> str: 
        return "Random Rot90 (multi axis)" 
    

class RandomFlip(object): 
    def __init__(self, dims: List[int] = [0, 1, 2]): 
        self.dims = dims 

    def __call__(self, mode1_image, mode2_image): 
        for dim in self.dims: 
            choice = np.random.choice([True, False]) 
            if choice: 
                mode1_image = np.flip(mode1_image, axis=dim) 
                mode2_image = np.flip(mode2_image, axis=dim) 
                
        return mode1_image, mode2_image 
    
    def __str__(self) -> str:
        return "Random Flip (multi axis)" 
    

class Normalize(object): 
    def __init__(self, mode: str = 'mm'): 
        assert mode in ('mm', 'ms') 
        self.mode = mode 

    @staticmethod 
    def _normalize(image, mode): 
        if mode == 'mm': 
            _min = np.min(image) 
            _max = np.max(image) 
            image = (image - _min) / (_max - _min) 
        else: 
            _mean = np.mean(image) 
            _std = np.std(image) 
            image = (image - _mean) / _std 
        return image 
    
    def __call__(self, mode1_image, mode2_image): 
        mode1_image = self._normalize(mode1_image, self.mode) 
        mode2_image = self._normalize(mode2_image, self.mode) 
        return mode1_image, mode2_image 
    
    def __str__(self) -> str:
        return "Normalize (max_min || mean_std)" 


class AdjustChannels(object): 
    def __init__(self): 
        super().__init__() 
    
    def __call__(self, mode1_image, mode2_image): 
        mode1_image = np.expand_dims(mode1_image, axis=0) 
        mode2_image = np.expand_dims(mode2_image, axis=0) 
        
        return mode1_image, mode2_image 
    
    def __str__(self) -> str:
        return "Adjust image and label's channels" 


class HalfResize(object): 
    def __init__(self): 
        super().__init__() 

    def __call__(self, mode1_image, mode2_image): 
        mode1_image = mode1_image[:, ::2, ::2, ::2] 
        mode2_image = mode2_image[:, ::2, ::2, ::2] 
        
        return mode1_image, mode2_image 

    def __str__(self) -> str:
        return "Half resize image and label" 
    

class TargetResize(object): 
    def __init__(self, target_shape: Tuple[int, int, int]): 
        self.target_shape = target_shape 
    
    def __call__(self, mode1_image, mode2_image): 
        mode1_image = T.resize(mode1_image[0], self.target_shape) 
        mode1_image = np.expand_dims(mode1_image, axis=0)  
        mode2_image = T.resize(mode2_image[0], self.target_shape) 
        mode2_image = np.expand_dims(mode2_image, axis=0) 

        return mode1_image, mode2_image  
    
    def __str__(self) -> str:
        return "Target resize image and label" 


class ToTensor(object): 
    def __init__(self): 
        super().__init__() 

    def __call__(self, mode1_image, mode2_image): 
        mode1_image = torch.from_numpy(mode1_image) 
        mode2_image = torch.from_numpy(mode2_image) 
        
        return mode1_image, mode2_image

    def __str__(self) -> str:
        return "ToTensor" 
    

class Compose(object): 
    def __init__(self, ops): 
        self.ops = ops 

    def __call__(self, mode1_image, mode2_image): 
        for op in self.ops: 
            if op != None: 
                mode1_image, mode2_image = op(mode1_image, mode2_image) 
        return mode1_image, mode2_image
    
    def __str__(self) -> str:
        return "Compose (multi operations)"  


def adjust_image(image: np.ndarray) -> np.ndarray: 
    """ Adjust dataloader's image to standard gray-level image. 

    Parameters: 
        image (Array): normalized image from dataloader [D, H, W] float32 0~1 

    Returns: 
        image (Array): gray level image to store [D, H, W] uint8 0~255 """

    # normalize first 
    _min, _max = np.min(image), np.max(image) 
    image = (image - _min) / (_max - _min) 
    image = image * 255 
    image = np.array(image, dtype='uint8') 

    return image 
    
