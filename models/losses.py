import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


def gaussian_kernel_3d(window_size: int, sigma: float, channel: int)->torch.tensor:

    if window_size % 2 == 0:
        raise ValueError("Kernel size must be odd.")

    x = torch.arange(-window_size // 2 + 1, window_size // 2 + 1)
    y = torch.arange(-window_size // 2 + 1, window_size // 2 + 1)
    z = torch.arange(-window_size // 2 + 1, window_size // 2 + 1)
    x, y, z = torch.meshgrid(x, y, z, indexing='xy')

    distance = x**2 + y**2 + z**2
    gaussian_kernel = torch.exp(-0.5 * (distance / (sigma**2)))
    gaussian_kernel /= torch.sum(gaussian_kernel)

    gaussian_kernel = gaussian_kernel.float().unsqueeze(0).unsqueeze(0)
    gaussian_kernel = Variable(gaussian_kernel.expand(channel, 1, window_size, window_size, window_size).contiguous())

    return gaussian_kernel


def _ssim(img1, img2, window, window_size, channel, size_average = True):
    mu1 = F.conv3d(img1, window, padding = window_size//2, groups = channel)
    mu2 = F.conv3d(img2, window, padding = window_size//2, groups = channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv3d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
    sigma2_sq = F.conv3d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
    sigma12 = F.conv3d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2                                  

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2)) 

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)
    

def ssim(img1, img2, fused, window_size = 11, sigma = 1.5, size_average = True):
    (_, channel, _, _, _) = img1.size()
    window = gaussian_kernel_3d(window_size, sigma, channel)
    
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return (_ssim(img1, fused, window, window_size, channel, size_average) + _ssim(img2, fused, window, window_size, channel, size_average)) / 2


def sobel_kernel(channel):
    hx = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    hy = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])

    hx = hx.float().unsqueeze(0).unsqueeze(0)
    hx = Variable(hx.expand(channel, 1, 3, 3).contiguous())
    hy = hy.float().unsqueeze(0).unsqueeze(0)
    hy = Variable(hy.expand(channel, 1, 3, 3).contiguous())

    return hx, hy


def _QABF(imgA, imgB, imgF, hx, hy, channel):
    Tg = 0.9994;
    kg = -15;
    Dg = 0.5;
    Ta = 0.9879;
    ka = -22;
    Da = 0.8;

    SAx = F.conv2d(imgA, hx, padding = 1, groups = channel) + 1e-10
    SAy = F.conv2d(imgA, hy, padding = 1, groups = channel)
    SBx = F.conv2d(imgB, hx, padding = 1, groups = channel) + 1e-10
    SBy = F.conv2d(imgB, hy, padding = 1, groups = channel)
    SFx = F.conv2d(imgF, hx, padding = 1, groups = channel) + 1e-10
    SFy = F.conv2d(imgF, hy, padding = 1, groups = channel)

    gA = torch.sqrt(SAx**2 + SAy**2)
    aA = torch.atan(SAy/SAx) * (SAx != 1e-10).float() + (torch.pi/2) * (SAx == 1e-10).float()
    gB = torch.sqrt(SBx**2 + SBy**2)
    aB = torch.atan(SBy/SBx) * (SBx != 1e-10).float() + (torch.pi/2) * (SBx == 1e-10).float()
    gF = torch.sqrt(SFx**2 + SFy**2)
    aF = torch.atan(SFy/SFx) * (SFx != 1e-10).float() + (torch.pi/2) * (SFx == 1e-10).float()

    gA = gA + 1e-30
    gF = gF + 1e-30
    gB = gB + 1e-30
    GAF = (gA/gF)*(gF>gA).float() + (gF/gA)*(gF<gA).float() + gF*(gF== gA).float()  # (0, 1)
    GBF = (gB/gF)*(gF>gB).float() + (gF/gB)*(gF<gB).float() + gF*(gF== gB).float()
    AAF = 1 - torch.abs(aA - aF) / (torch.pi/2)  # (-1, 1)
    ABF = 1 - torch.abs(aB - aF) / (torch.pi/2)

    QgAF = Tg / (1 + torch.exp(kg * (GAF - Dg)))  # (0, 1)
    QaAF = Ta / (1 + torch.exp(ka * (AAF - Da)))  # (0, 1)
    QAF = QgAF * QaAF
    QgBF = Tg / (1 + torch.exp(kg * (GBF - Dg)))  # (0, 1)
    QaBF = Ta / (1 + torch.exp(ka * (ABF - Da)))  # (0, 1)
    QBF = QgBF * QaBF

    QABF = torch.sum((QAF*gA + QBF*gB), dim=(1,2,3)) / torch.sum((gA + gB), dim=(1,2,3))
    
    return QABF.mean()


def QABF(imgA: torch.tensor, imgB: torch.tensor, imgF: torch.tensor):
    B, C, D, H, W = imgA.size()
    imgA = imgA.permute(0,2,1,3,4).reshape(B*D, C, H, W) 
    imgB = imgB.permute(0,2,1,3,4).reshape(B*D, C, H, W) 
    imgF = imgF.permute(0,2,1,3,4).reshape(B*D, C, H, W) 
    hx, hy = sobel_kernel(C)

    if imgA.is_cuda:
        hx = hx.cuda(imgA.get_device())
        hy = hy.cuda(imgA.get_device())
    hx = hx.type_as(imgA)
    hy = hy.type_as(imgA)

    return _QABF(imgA, imgB, imgF, hx, hy, C)


def _NABF(imgA, imgB, imgF, hx, hy, channel):
    Tg = 0.9999;
    kg = -19;
    Dg = 0.5;
    Ta = 0.9995;
    ka = -22;
    Da = 0.5;
    Lg = 1.5;
    Td = 2;
    wt_min = 0.001;

    SAx = F.conv2d(imgA, hx, padding = 1, groups = channel) + 1e-10
    SAy = F.conv2d(imgA, hy, padding = 1, groups = channel)
    SBx = F.conv2d(imgB, hx, padding = 1, groups = channel) + 1e-10
    SBy = F.conv2d(imgB, hy, padding = 1, groups = channel)
    SFx = F.conv2d(imgF, hx, padding = 1, groups = channel) + 1e-10
    SFy = F.conv2d(imgF, hy, padding = 1, groups = channel)

    gA = torch.sqrt(SAx**2 + SAy**2)
    aA = torch.atan(SAy/SAx) * (SAx != 1e-10).float() + (torch.pi/2) * (SAx == 1e-10).float()
    gB = torch.sqrt(SBx**2 + SBy**2)
    aB = torch.atan(SBy/SBx) * (SBx != 1e-10).float() + (torch.pi/2) * (SBx == 1e-10).float()
    gF = torch.sqrt(SFx**2 + SFy**2)
    aF = torch.atan(SFy/SFx) * (SFx != 1e-10).float() + (torch.pi/2) * (SFx == 1e-10).float()

    gA = gA + 1e-30
    gF = gF + 1e-30
    gB = gB + 1e-30
    GAF = (gA/gF)*(gF>gA).float() + (gF/gA)*(gF<gA).float() + gF*(gF== gA).float()  # (0, 1)
    GBF = (gB/gF)*(gF>gB).float() + (gF/gB)*(gF<gB).float() + gF*(gF== gB).float()
    AAF = 1 - torch.abs(aA - aF) / (torch.pi/2)  # (-1, 1)
    ABF = 1 - torch.abs(aB - aF) / (torch.pi/2)

    QgAF = Tg / (1 + torch.exp(kg * (GAF - Dg)))  # (0, 1)
    QaAF = Ta / (1 + torch.exp(ka * (AAF - Da)))  # (0, 1)
    QAF = QgAF * QaAF
    QgBF = Tg / (1 + torch.exp(kg * (GBF - Dg)))  # (0, 1)
    QaBF = Ta / (1 + torch.exp(ka * (ABF - Da)))  # (0, 1)
    QBF = QgBF * QaBF

    wtA = torch.where(gA >= Td, gA ** Lg, wt_min)
    wtB = torch.where(gB >= Td, gB ** Lg, wt_min)
    na = torch.where((gF > gA) & (gF > gB), 1, 0)

    NABF = torch.sum(na * ((1 - QAF) * wtA + (1 - QBF) * wtB), dim=(1,2,3)) / torch.sum(wtA + wtB, dim=(1,2,3))
    
    return NABF.mean()


def NABF(imgA: torch.tensor, imgB: torch.tensor, imgF: torch.tensor):
    B, C, D, H, W = imgA.size()
    imgA = imgA.permute(0,2,1,3,4).reshape(B*D, C, H, W) 
    imgB = imgB.permute(0,2,1,3,4).reshape(B*D, C, H, W) 
    imgF = imgF.permute(0,2,1,3,4).reshape(B*D, C, H, W) 
    hx, hy = sobel_kernel(C)

    if imgA.is_cuda:
        hx = hx.cuda(imgA.get_device())
        hy = hy.cuda(imgA.get_device())
    hx = hx.type_as(imgA)
    hy = hy.type_as(imgA)

    return _NABF(imgA, imgB, imgF, hx, hy, C)


def marginalPdf(sigma, bins, values):

    residuals = values - bins.unsqueeze(0).unsqueeze(0)
    kernel_values = torch.exp(-0.5*(residuals / sigma).pow(2))

    pdf = torch.mean(kernel_values, dim=1)
    normalization = torch.sum(pdf, dim=1).unsqueeze(1) + 1e-10
    pdf = pdf / normalization

    return pdf, kernel_values


def jointPdf(kernel_values1, kernel_values2):

    joint_kernel_values = torch.matmul(kernel_values1.transpose(1, 2), kernel_values2) 
    normalization = torch.sum(joint_kernel_values, dim=(1,2)).view(-1, 1, 1) + 1e-10
    pdf = joint_kernel_values / normalization

    return pdf


def Norm255(x):
    minvalue = torch.min(x)
    maxvalue = torch.max(x)
    out = (x - minvalue) / (maxvalue - minvalue) * 255

    return out


def _Get_MI(input1, input2, sigma, bins)->torch.tensor:

    input1 = Norm255(input1)
    input2 = Norm255(input2)

    B, C, H, W = input1.shape
    assert((input1.shape == input2.shape))

    x1 = input1.permute(0,2,3,1).view(B, H*W, C)
    x2 = input2.permute(0,2,3,1).view(B, H*W, C)

    pdf_x1, kernel_values1 = marginalPdf(sigma, bins, x1)
    pdf_x2, kernel_values2 = marginalPdf(sigma, bins, x2)
    pdf_x1x2 = jointPdf(kernel_values1, kernel_values2)

    H_x1 = -torch.sum(pdf_x1*torch.log(pdf_x1 + 1e-10), dim=1)
    H_x2 = -torch.sum(pdf_x2*torch.log(pdf_x2 + 1e-10), dim=1)
    H_x1x2 = -torch.sum(pdf_x1x2*torch.log(pdf_x1x2 + 1e-10), dim=(1,2))
    
    mutual_information = H_x1 + H_x2 - H_x1x2
    mutual_information = (2*mutual_information + 1e-10) / (H_x1 + H_x2 + 1e-10)

    return mutual_information.mean()


def MI(x1, x2, F)->torch.tensor:
    B, C, D, H, W = x1.size()
    x1 = x1.permute(0,2,1,3,4).reshape(B*D, C, H, W) 
    x2 = x2.permute(0,2,1,3,4).reshape(B*D, C, H, W) 
    F = F.permute(0,2,1,3,4).reshape(B*D, C, H, W)

    sigma=0.1
    num_bins = 128
    bins = nn.Parameter(torch.linspace(0, 255, num_bins).float(), requires_grad=False)
    if x1.is_cuda:
        bins = bins.cuda(x1.get_device())

    return (_Get_MI(x1, F, sigma, bins) + _Get_MI(x2, F, sigma, bins)) / 2


def corr2(a, b):
    a = a - torch.mean(a, dim=(1,2,3,4), keepdim=True)
    b = b - torch.mean(b, dim=(1,2,3,4), keepdim=True)
    r = (torch.sum(a * b, dim=(1,2,3,4)) + 1e-10) / (torch.sqrt(torch.sum(a * a, dim=(1,2,3,4)) * torch.sum(b * b, dim=(1,2,3,4))) + 1e-10)
    return r.mean()


def SCD(A, B, F):

    r = (corr2(F - B, A) + corr2(F - A, B)) / 2

    return r


def MSE(img1, img2, fused):

    return (F.mse_loss(img1, fused, reduction='mean') + F.mse_loss(img2, fused, reduction='mean')) / 2


def _vif(img1, img2, channel):
    sigma_nsq = 2
    eps = 1e-10

    num = torch.tensor(0.0).cuda()
    den = torch.tensor(0.0).cuda()

    for scale in range(1, 5):
        N = 2 ** (4 - scale + 1) + 1
        sd = N / 5.0
        kernel = gaussian_kernel_3d(N, sd, channel)
        if img1.is_cuda:
            kernel = kernel.cuda(img1.get_device())

        if (scale > 1):
            img1 = F.conv3d(img1, kernel, groups = channel)
            img2 = F.conv3d(img2, kernel, groups = channel)
            img1 = img1[:, :, ::2, ::2, ::2]
            img2 = img2[:, :, ::2, ::2, ::2]

        mu1 = F.conv3d(img1, kernel, groups = channel)
        mu2 = F.conv3d(img2, kernel, groups = channel)
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        sigma1_sq = F.conv3d(img1*img1, kernel, groups = channel) - mu1_sq
        sigma2_sq = F.conv3d(img2*img2, kernel, groups = channel) - mu2_sq
        sigma12   = F.conv3d(img1*img2, kernel, groups = channel) - mu1_mu2

        sigma1_sq[sigma1_sq < 0] = 0
        sigma2_sq[sigma2_sq < 0] = 0

        g = sigma12 / (sigma1_sq + eps)
        sv_sq = sigma2_sq - g * sigma12

        g = torch.where(sigma1_sq < eps, 0, g) 
        sv_sq = torch.where(sigma1_sq < eps, sigma2_sq, sv_sq)
        sigma1_sq = torch.where(sigma1_sq < eps, 0, sigma1_sq) 

        g = torch.where(sigma2_sq < eps, 0, g)
        sv_sq = torch.where(sigma2_sq < eps, 0, sv_sq)

        sv_sq = torch.where(g < 0, sigma2_sq, sv_sq)
        g = torch.where(g < 0, 0, g)
        sv_sq = torch.where(sv_sq <= eps, eps, sv_sq)

        num += torch.sum(torch.log10(1 + g * g * sigma1_sq / (sv_sq + sigma_nsq)))
        den += torch.sum(torch.log10(1 + sigma1_sq / sigma_nsq))
    
    vifq = (num + eps) / (den + eps)

    return vifq

    
def VIF(img1, img2, fused):
    B, C, D, H, W = img1.shape

    return (_vif(img1, fused, C) + _vif(img2, fused, C)) / 2


def AG(img1, img2, image):
    image = Norm255(image)
    dx = image[:,:,1:,:-1,:-1] - image[:,:,:-1,:-1,:-1] 
    dy = image[:,:,:-1,1:,:-1] - image[:,:,:-1,:-1,:-1] 
    dz = image[:,:,:-1,:-1,1:] - image[:,:,:-1,:-1,:-1] 
    ds = torch.sqrt((dx**2 + dy**2 + dz**2) / 3)
    ag = torch.mean(ds)

    return ag


def _psnr(img1, img2, maxvalue = 1):
    mse = torch.mean((img1 - img2) ** 2)
    return 10 * torch.log10((maxvalue ** 2) / mse)


def PSNR(img1, img2, fused):
    return (_psnr(img1, fused) + _psnr(img2, fused)) / 2


def EN(img1, img2, fused):
    fused = Norm255(fused)
    histogram = torch.histc(fused, bins=255, min=0, max=255)
    histogram = histogram / histogram.sum()

    entropy = -(histogram * torch.log2(histogram + 1e-7)).sum()
    return entropy


def SD(img1, img2, fused):
    u = torch.mean(fused)
    SD = torch.sqrt(((fused - u)**2).mean())
    return SD


def SF(img1, img2, fused):
    F1 = (torch.diff(fused, dim=2)**2).mean()
    F2 = (torch.diff(fused, dim=3)**2).mean()
    F3 = (torch.diff(fused, dim=4)**2).mean()
    SF = torch.sqrt(F1 + F2 + F3)
    return SF

def rSFe(img1, img2, fused):
    SFf = SF(img1, img2, fused)

    Fr1 = (torch.max(torch.abs(torch.diff(img1, dim=2)), torch.abs(torch.diff(img2, dim=2)))**2).mean()
    Fr2 = (torch.max(torch.abs(torch.diff(img1, dim=3)), torch.abs(torch.diff(img2, dim=3)))**2).mean()
    Fr3 = (torch.max(torch.abs(torch.diff(img1, dim=4)), torch.abs(torch.diff(img2, dim=4)))**2).mean()
    SFr = torch.sqrt(Fr1 + Fr2 + Fr3)

    rSFe = (SFf - SFr) / SFr

    return rSFe


losses_mapping_dict = {'mse' : MSE,
                       'ssim': ssim,
                       'QABF': QABF,
                       'MI'  : MI,
                       'SCD' : SCD,
                       'VIF' : VIF,
                       'AG'  : AG,
                       'PSNR': PSNR,
                       'EN'  : EN,
                       'SD'  : SD,
                       'SF'  : SF,
                       'NABF': NABF,
                       'rSFe': rSFe,}


def Get_loss(loss_list: list, lamdas: list, img1: torch.tensor, img2: torch.tensor, fused: torch.tensor,
             extra_out: list = None, extra_loss: list = None)->dict:
    loss_t = torch.tensor(0.0).cuda()
    losses = {}

    for (loss, lamda) in zip(loss_list, lamdas):
        if loss in ['ssim', 'QABF', 'MI', 'SCD', 'VIF', 'AG', 'PSNR']:
            loss_v = 1 - losses_mapping_dict[loss](img1, img2, fused)
        else:
            loss_v = losses_mapping_dict[loss](img1, img2, fused)
        loss_v *= lamda

        loss_t += loss_v
        losses.update({loss: loss_v})

    if extra_out is not None:
        if extra_out[0] is not None:
            for out, loss in zip(extra_out, extra_loss):
                if loss in ['ssim', 'QABF', 'MI', 'SCD', 'VIF', 'AG', 'PSNR']:
                    loss_v = 1 - losses_mapping_dict[loss](img1, img2, out)
                else:
                    loss_v = losses_mapping_dict[loss](img1, img2, out)

            loss_t += loss_v

    return loss_t, losses


def Get_metric(metric_list: list, img1: torch.tensor, img2: torch.tensor, fused: torch.tensor)->dict:
    metrics = {}

    for metric in metric_list: 
        metric_v = losses_mapping_dict[metric](img1, img2, fused)
        metrics.update({metric: metric_v.item()})

    return metrics


if __name__ == '__main__':
    from dataloaders import get_dataloader
    import argparse

    parser = argparse.ArgumentParser() 
    # add dataset related arguments 
    parser.add_argument('--batch_size', type=int, default=2, help='the number of samples in a batch') 
    parser.add_argument('--dataroot', type=str, default='./FusionDataset', help='root dir where stores data') 
    parser.add_argument('--root_dir', type=str, default="", help='specified dir to store data')
    parser.add_argument('--which_set', type=str, default="brats2020", help='used dataset') 
    parser.add_argument('--is_resize', default=True, action='store_true') 
    parser.add_argument('--target_shape', type=str, default='128,196,196', help='target shape used in RT TargetResize')  
    parser.add_argument('--mode1', type=str, default='t1', help='which mode to fuse') 
    parser.add_argument('--mode2', type=str, default='t2', help='which mode to fuse') 
    args = parser.parse_args() 

    train_loader, test_loader = get_dataloader(args) 
    for step, (mode1_image, mode2_image) in enumerate(train_loader): 
        mode1_image = mode1_image.cuda()
        mode2_image = mode2_image.cuda()
        print(mode1_image.shape)

        # print(MI(mode1_image, mode2_image, (mode1_image + mode2_image)/2))
        # print(ssim(mode1_image, mode2_image, (mode1_image + mode2_image)/2))
        # print(QABF(mode1_image, mode2_image, (mode1_image + mode2_image)/2))
        # print(SCD(mode1_image, mode2_image, (mode1_image + mode2_image)/2))
        # print(VIF(mode1_image, mode2_image, (mode1_image + mode2_image)/2))
        # print(AG(mode1_image, mode2_image, (mode1_image + mode2_image)/2))
        # print(PSNR(mode1_image, mode2_image, (mode1_image + mode2_image)/2))
        # print(EN(mode1_image, mode2_image, (mode1_image + mode2_image)/2))
        # print(SD(mode1_image, mode2_image, (mode1_image + mode2_image)/2))
        print(SF(mode1_image, mode2_image, (mode1_image + mode2_image)/2))

        break



