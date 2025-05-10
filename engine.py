import os
import json
import numpy as np
import math
import torch
import torch.nn.functional as F
import SimpleITK as sitk
import pandas as pd 

import models.losses as Losses
from utils import utils


def train(args, train_loader, test_loader, model, device, logger):

    optimizer = torch.optim.Adam(model.parameters(), args.lr, betas=(0.9, 0.999), weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.max_epoch)

    iter_num = 0
    best_metric = 0.0 

    for epoch in range(args.max_epoch): 
        for step, (mode1_image, mode2_image) in enumerate(train_loader): 
            mode1_image = mode1_image.to(device) 
            mode2_image = mode2_image.to(device) 

            fused_image, base, detail, rec_image = model(mode1_image, mode2_image) 

            train_loss_, losses = Losses.Get_loss(['mse', 'ssim'], [args.alpha, args.beta],
                                                 mode1_image, mode2_image, fused_image,
                                                 [base, detail], ['mse', 'ssim'])
            # ***************************
            lamda = math.pow(math.e, (step - args.max_epoch) * 0.002)
            gamma = lamda * args.gamma
            loss_SF = gamma * -Losses.SF(mode1_image, mode2_image, fused_image)
            loss_rec = F.mse_loss(rec_image, torch.cat([mode1_image, mode2_image], dim=1), reduction='mean') 

            train_loss = train_loss_ + loss_SF + loss_rec
            losses.update({'SF': loss_SF})
            losses.update({'rec': loss_rec})
            # ***************************
            metrics = Losses.Get_metric(['ssim', 'QABF', 'SF'], mode1_image, mode2_image, fused_image)

            optimizer.zero_grad() 
            train_loss.backward() 
            optimizer.step() 

            iter_num += 1 
            
            # log the losses
            message = "Train: " 
            message += "iteration: {}, ".format(iter_num) 
            for loss_name, loss_value in losses.items(): 
                message += "{}_loss: {:.3f} ".format(loss_name, loss_value) 
            for metric_name, metric_value in metrics.items():
                message += "{}: {:.3f} ".format(metric_name, metric_value)   
            logger.info(message) 

            # validate 
            if iter_num >= args.save_freq and iter_num % args.save_freq == 0: 
                # validate 
                val_metric = validate(args, test_loader, model, device, logger, iter_num) 

                if val_metric > best_metric:
                    best_metric = val_metric
                    torch.save(model.state_dict(), os.path.join(args.output_dir, 'best_model.pth') ) 
                # save latest model 
                save_model_path = os.path.join(args.output_dir, 'iter_{}_metric_{:.3f}.pth'.format(iter_num, val_metric))
                torch.save(model.state_dict(), save_model_path) 

                model.train() 

            if iter_num == args.max_iters: 
                logger.info("end training") 
                return 0 
                
        lr_scheduler.step()


def validate(args, test_loader, model, device, logger, iter_num):

    ssim = utils.AverageMeter()

    model.eval()
    with torch.no_grad():
        for step, (mode1_image, mode2_image) in enumerate(test_loader): 
            mode1_image = mode1_image.to(device) 
            mode2_image = mode2_image.to(device)
            N = mode1_image.size(0) # batch size 

            fused_image, _, _, _ = model(mode1_image, mode2_image) 
            metrics = Losses.Get_metric(['ssim', 'QABF', 'SF'], mode1_image, mode2_image, fused_image) 

            ssim.update(metrics['ssim'], N)
            
            if step >= args.max_samples - 1:
                break 

    message = "Valid: " 
    message += "iteration: {}, ".format(iter_num) 
    for metric_name, metric_value in metrics.items():
        message += "{}: {:.3f} ".format(metric_name, metric_value)    
    logger.info(message)  

    return ssim.avg


@torch.no_grad() 
def evaluate(args, test_loader, model, device):
    metric_dict = {'ssim': [], 
                   'QABF': [],
                   'MI'  : [],
                   'SCD' : [],
                   'VIF' : [],
                   'AG'  : [],
                   'PSNR': [],
                   'EN'  : [],
                   'SD'  : [],
                   'SF'  : [],
                   'NABF': [],
                   'rSFe': [],}

    for step, (mode1_image, mode2_image) in enumerate(test_loader): 
        mode1_image = mode1_image.to(device) 
        mode2_image = mode2_image.to(device)

        N = mode1_image.size(0) # batch size 
        assert N == 1, "Batch size must be 1 during evaluation"

        fused_image, _, _, _ = model(mode1_image, mode2_image) 
        metrics = Losses.Get_metric(['ssim', 'QABF', 'MI', 'SCD', 'VIF', 'AG', 'PSNR', 'EN', 'SD', 'SF', 'NABF', 'rSFe'], 
                                    mode1_image, mode2_image, fused_image) 

        if not os.path.exists(os.path.join(args.output_dir, "samples")):
            os.mkdir(os.path.join(args.output_dir, "samples"))
        save_dir = os.path.join(args.output_dir, "samples", "sample_{}".format(step)) 
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        with open(os.path.join(save_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f) 
        f.close() 
        mode1_image = utils.Renorm_image(mode1_image)
        sitk.WriteImage(sitk.GetImageFromArray(mode1_image), os.path.join(save_dir, "mode1_image.nii.gz"))
        mode2_image = utils.Renorm_image(mode2_image)
        sitk.WriteImage(sitk.GetImageFromArray(mode2_image), os.path.join(save_dir, "mode2_image.nii.gz"))
        fused_image = utils.Renorm_image(fused_image)
        sitk.WriteImage(sitk.GetImageFromArray(fused_image), os.path.join(save_dir, "fused_image.nii.gz"))

        for metric_name, metric_value in metrics.items():
            metric_dict[metric_name].append(metric_value)

        print("sample_{} have done".format(step))
        if step >= 49:
            break

    for metric_name, metric_list in metric_dict.items():
        metric_dict[metric_name].append(np.mean(metric_list))
        metric_dict[metric_name].append(np.std(metric_list))
    metric_df = pd.DataFrame(metric_dict) 
    metric_df.to_excel(os.path.join('./results', '{}.xlsx'.format(args.exp_name)), index=False) 