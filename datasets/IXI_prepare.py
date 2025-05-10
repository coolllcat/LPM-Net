import os 
import numpy as np
import cv2
import SimpleITK as sitk


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
    resampler.SetSize(Size if Size is not None else vol.GetSize())
    newvol = resampler.Execute(vol)
    
    return newvol


def IDScale(Image: sitk.Image) -> sitk.Image:
    Image_volume = sitk.GetArrayFromImage(Image)
    Image_volume = Image_volume / Image_volume.max() * 255
    Image_new = sitk.GetImageFromArray(Image_volume)
    Image_new.CopyInformation(Image)

    return Image_new


files = os.listdir('./IXI_Source/IXI-T1')

for file in files:
    T1_path = os.path.join('./IXI_Source/IXI-T1', file)
    T2_path = os.path.join('./IXI_Source/IXI-T2', file.replace('T1', 'T2'))
    PD_path = os.path.join('./IXI_Source/IXI-PD', file.replace('T1', 'PD'))

    T1_Image = sitk.ReadImage(T1_path)
    try:
        T2_Image = sitk.ReadImage(T2_path)
        PD_Image = sitk.ReadImage(PD_path)
    except:
        continue

    T2_Spacing = T2_Image.GetSpacing()
    T2_Image_new = resamplevolume(T2_Image, Spacing=[T2_Spacing[0], T2_Spacing[1], T2_Spacing[1]],
                                        Origin=None,
                                        Direction=[1.0,0.0,0.0,0.0,-1.0,0.0,0.0,0.0,1.0],
                                        Size=[256,256,192])
    T1_Image_new = resamplevolume(T1_Image, Spacing=T2_Image_new.GetSpacing(),
                                        Origin=T2_Image_new.GetOrigin(),
                                        Direction=T2_Image_new.GetDirection(),
                                        Size=T2_Image_new.GetSize())
    PD_Image_new = resamplevolume(PD_Image, Spacing=T2_Image_new.GetSpacing(),
                                        Origin=T2_Image_new.GetOrigin(),
                                        Direction=T2_Image_new.GetDirection(),
                                        Size=T2_Image_new.GetSize())
    
    T2_Image_new = IDScale(T2_Image_new)
    T1_Image_new = IDScale(T1_Image_new)
    PD_Image_new = IDScale(PD_Image_new)
    
    save_dir = os.path.join('./IXI-T1-T2-PD', file[:-10])
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    sitk.WriteImage(T2_Image_new, os.path.join(save_dir, 'T2.nii.gz'))
    sitk.WriteImage(T1_Image_new, os.path.join(save_dir, 'T1.nii.gz'))
    sitk.WriteImage(PD_Image_new, os.path.join(save_dir, 'PD.nii.gz'))

    print('{} has done...'.format(file[:-10]))