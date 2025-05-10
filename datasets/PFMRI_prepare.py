
import SimpleITK as sitk
import os
import shutil
import cv2
import numpy as np


def resamplevolume(vol: sitk.Image, Spacing:list =None, Origin: list=None, Direction: list=None, Size: list=None):
    inputsize = vol.GetSize()
    inputSpacing = vol.GetSpacing()
    outputsize = [inputsize[0]*inputSpacing[0]/Spacing[0], inputsize[1]*inputSpacing[1]/Spacing[1], inputsize[2]*inputSpacing[2]/Spacing[2]]
    
    transform = sitk.Transform()
    transform.SetIdentity()

    resampler = sitk.ResampleImageFilter()
    resampler.SetTransform(transform)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetOutputOrigin(Origin if Origin is not None else vol.GetOrigin())
    resampler.SetOutputDirection(Direction if Direction is not None else vol.GetDirection())
    resampler.SetOutputSpacing(Spacing if Spacing is not None else vol.GetSpacing())
    resampler.SetSize(Size if Size is not None else outputsize)
    newvol = resampler.Execute(vol)
    
    return newvol


def IDScale(Image: sitk.Image) -> sitk.Image:
    Image_volume = sitk.GetArrayFromImage(Image)
    Image_volume = Image_volume / Image_volume.max() * 255
    Image_new = sitk.GetImageFromArray(Image_volume)
    Image_new.CopyInformation(Image)

    return Image_new


def sitk_dcms_read(dcm_dir):
    reader = sitk.ImageSeriesReader()
    seriesIDs = reader.GetGDCMSeriesIDs(dcm_dir)
    dcm_series = reader.GetGDCMSeriesFileNames(dcm_dir, seriesIDs[0])
    if len(dcm_series) > 1:
        reader.SetFileNames(dcm_series)
        itkimage = reader.Execute()
    else:
        itkimage = sitk.ReadImage(dcm_series[0])
    #numpyImage = sitk.GetArrayFromImage(itkimage)
    #numpyOrigin = list(reversed(itkimage.GetOrigin()))
    #numpySpacing = list(reversed(itkimage.GetSpacing()))
    #numpyDirection = list(reversed(itkimage.GetDirection()))
        
    return itkimage #numpyImage, numpyOrigin, numpySpacing, numpyDirection


def List_check(inputs: list, checked: str):
    for inputname in inputs:
        if checked in inputname:
            return inputname
    return None


rootdir = './Prostate-Fused-MRI-Source'
dirs = os.listdir(rootdir)

for sdir in dirs:
    work_dir = os.path.join(rootdir, sdir)
    
    work_dir = os.path.join(work_dir, os.listdir(work_dir)[0])
    
    image_series = os.listdir(work_dir)
    
    if (List_check(image_series, 'T1 AXIAL SM FOV') is not None) & (List_check(image_series, 'T2 AXIAL SM FOV') is not None):
        
        T1_path = os.path.join(work_dir, List_check(image_series, 'T1 AXIAL SM FOV'))
        T2_path = os.path.join(work_dir, List_check(image_series, 'T2 AXIAL SM FOV'))
        
        T1_Image = sitk_dcms_read(T1_path)
        T1_Image = resamplevolume(T1_Image, Spacing=[0.5, 0.5, 0.5],
                                     Origin=None,
                                     Direction=None, #[1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0]
                                     Size=[256,256,160])
        T1_Image.SetDirection([1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0])
        T1_Image = IDScale(T1_Image)
        T2_Image = sitk_dcms_read(T2_path)
        T2_Image = resamplevolume(T2_Image, Spacing=[0.5, 0.5, 0.5],
                                     Origin=None,
                                     Direction=None, #[1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0]
                                     Size=[256,256,160])
        T2_Image.SetDirection([1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0])
        T2_Image = IDScale(T2_Image)
        
        save_dir = os.path.join('./Prostate-Fused-MRI', '{}_T1_T2'.format(sdir))
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        sitk.WriteImage(T1_Image, os.path.join(save_dir, 'mode1.nii.gz'))
        sitk.WriteImage(T2_Image, os.path.join(save_dir, 'mode2.nii.gz'))

    if List_check(image_series, 'T1 VIBE OPP IN  PHASE AXIAL MIN TE 1 AND 2') is not None:

        path = os.path.join(work_dir, List_check(image_series, 'T1 VIBE OPP IN  PHASE AXIAL MIN TE 1 AND 2'))
        
        reader = sitk.ImageSeriesReader()
        seriesIDs = reader.GetGDCMSeriesIDs(path)
        series = reader.GetGDCMSeriesFileNames(path, seriesIDs[0])
        
        reader.SetFileNames(series[:len(series)//2])
        Image1 = reader.Execute()
        reader.SetFileNames(series[len(series)//2:])
        Image2 = reader.Execute()
        
        Image1_Spacing = Image1.GetSpacing()
        Image1 = resamplevolume(Image1, Spacing=[Image1_Spacing[0], Image1_Spacing[1], Image1_Spacing[1]],
                                     Origin=None,
                                     Direction=[1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0],
                                     Size=[256,256,160])
        Image1 = IDScale(Image1)
        Image2_Spacing = Image2.GetSpacing()
        Image2 = resamplevolume(Image2, Spacing=[Image2_Spacing[0], Image2_Spacing[1], Image2_Spacing[1]],
                                     Origin=None,
                                     Direction=[1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0],
                                     Size=[256,256,160])
        Image2 = IDScale(Image2)
        
        save_dir = os.path.join('./Prostate-Fused-MRI', '{}_T1_T1'.format(sdir))
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        sitk.WriteImage(Image1, os.path.join(save_dir, 'mode1.nii.gz'))
        sitk.WriteImage(Image2, os.path.join(save_dir, 'mode2.nii.gz'))
























