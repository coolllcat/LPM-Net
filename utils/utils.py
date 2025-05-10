import os 
import time
import logging 
import torch
import numpy as np


def get_logger(file_path):
    """ Make python logger """
    logger = logging.getLogger('darts')
    log_format = '%(asctime)s | %(message)s'
    formatter = logging.Formatter(log_format, datefmt='%m/%d %I:%M:%S %p')
    file_handler = logging.FileHandler(file_path)
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.setLevel(logging.INFO)

    return logger


class AverageMeter():
    """ Computes and stores the average and current value """
    def __init__(self):
        self.reset()

    def reset(self):
        """ Reset all statistics """
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """ Update statistics """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

        
def Renorm_image(image: torch.tensor)->np.ndarray:
    image = image.squeeze().cpu().numpy()
    
    _min, _max = np.min(image), np.max(image) 
    image = (image - _min) / (_max - _min) 
    image = image * 255 
    image = np.array(image, dtype='uint8') 

    return image