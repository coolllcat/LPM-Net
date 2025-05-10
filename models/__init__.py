import os
from models.Ours_mask import PMLNet


def get_model(args):
    
    if args.which_model == 'Ours':
        model = PMLNet(args)

    return model