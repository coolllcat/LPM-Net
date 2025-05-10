import os 
import numpy as np
import torch
import torch.nn as nn
import argparse

from dataloaders import get_dataloader
from models import get_model
from utils import utils
from engine import train, evaluate

from setproctitle import setproctitle 


def main(args):
    print(args) 
    # init steps
    device = torch.device(args.device) 
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed) 
    torch.backends.cudnn.benchmark = True 

    # set dataset and models
    train_loader, test_loader = get_dataloader(args) 
    args.max_epoch = args.max_iters // len(train_loader) + 1 
    model = get_model(args)
    num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad) 
    print('model number of parameters: {:.3f} MB'.format(num_parameters / 1e6))  

    setproctitle(args.exp_name) 
    args.output_dir = os.path.join(args.checkpoints_dir, args.exp_name) 

    if not args.eval:
        if not os.path.exists(args.output_dir):
            os.mkdir(args.output_dir)
        if os.path.exists(os.path.join(args.output_dir, "train.log")): 
            os.remove(os.path.join(args.output_dir, "train.log")) 
        logger = utils.get_logger(os.path.join(args.output_dir, "train.log")) 
        logger.info("Logger is set - training start") 
        if len(args.gpus) > 1:
            model = nn.DataParallel(model)
        model.to(device)
        train(args, train_loader, test_loader, model, device, logger)

    else:
        samples_dir = os.path.join(args.output_dir, 'samples') 
        if not os.path.join(samples_dir): 
            os.makedirs(samples_dir) 
        model.load_state_dict({k.replace('module.',''):v for k,v in torch.load(os.path.join(args.output_dir, args.eval_cp)).items()}) 
        model.to(device) 
        evaluate(args, test_loader, model, device)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser() 
    # add dataset related arguments 
    parser.add_argument('--batch_size', type=int, default=1, help='the number of samples in a batch') 
    parser.add_argument('--dataroot', type=str, default='./datasets', help='root dir where stores data') 
    parser.add_argument('--root_dir', type=str, default="", help='specified dir to store data')
    parser.add_argument('--which_set', type=str, default="", help='used dataset') 
    parser.add_argument('--is_resize', action='store_true') 
    parser.add_argument('--target_shape', type=str, help='target shape used in RT TargetResize')  
    parser.add_argument('--mode1', type=str, help='which mode to fuse') 
    parser.add_argument('--mode2', type=str, help='which mode to fuse') 

    # add train related arguments 
    parser.add_argument('--which_model', type=str, default="", help='used model') 
    parser.add_argument('--lr', type=float, default=1e-4, help='lr for weights')
    parser.add_argument('--gpus', type=str,  default='0', help='GPU to use')
    parser.add_argument('--seed', type=int, default=2, help='random seed')
    parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay') 
    parser.add_argument('--device', type=str, default='cuda:0') 
    parser.add_argument('--max_iters', type=int, default=3000, help='maximum number of iterations to train') 
    parser.add_argument('--max_epoch', type=int, default=1, help='maximum number of epochs to train') 

    # add model related arguments
    parser.add_argument('--base_c', type=int, default=8)
    parser.add_argument('--depth_c', type=int, default=2)
    parser.add_argument('--depth_m', type=int, default=2)
    parser.add_argument('--iscfc', type=str2bool, default=True)
    parser.add_argument('--fuse_method', type=str, default='max')
    parser.add_argument('--alpha', type=float, default=0.05)
    parser.add_argument('--beta', type=float, default=1)
    parser.add_argument('--gamma', type=float, default=0.5)

    # add save related arguments
    parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints') 
    parser.add_argument('--eval', action='store_true', default=False) 
    parser.add_argument('--eval_cp', default='best_model.pth') 
    parser.add_argument('--save_freq', type=int, default=200, help='validate the model per save_freq iterations') 
    parser.add_argument('--max_samples', type=int, default=20, help='maximum number of samples to be evaluated')
    parser.add_argument('--reuse', type=str, default=None) 
    parser.add_argument('--rec', type=float, default=0.2)

    args = parser.parse_args() 
    args.gpus = [int(s) for s in args.gpus.split(',')]
    args.exp_name = "{}_{}_{}_{}_c{}_{}_{}_{}_{}_{}_{}_{}".format(args.which_model, args.which_set.upper(), args.mode1.upper(), args.mode2.upper(),
                                                              args.base_c, args.depth_c, args.depth_m, args.fuse_method,
                                                              'iscfc' if args.iscfc else 'nocfc', args.alpha, args.beta, args.gamma)
    if args.reuse is not None:
        args.exp_name += '_reuse{}'.format(args.reuse)
    if args.rec > 0:
        args.exp_name += '_rec{}'.format(args.rec)

    main(args)