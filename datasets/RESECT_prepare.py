import os 
import numpy as np
import cv2
import SimpleITK as sitk


def resamplevolume(vol, Origin=None, Direction=None):
    
    transform = sitk.Transform()
    transform.SetIdentity()

    resampler = sitk.ResampleImageFilter()
    resampler.SetTransform(transform)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetOutputOrigin(Origin if Origin is not None else vol.GetOrigin())
    resampler.SetOutputDirection(Direction if Direction is not None else vol.GetDirection())
    resampler.SetOutputSpacing(vol.GetSpacing())
    resampler.SetSize(vol.GetSize())
    newvol = resampler.Execute(vol)
    
    return newvol


dirs = os.listdir('./RESECT_Source')
for path_dir in dirs:
    T1_path = os.path.join('./RESECT_Source', path_dir, 'MRI', '{}-T1.nii.gz'.format(path_dir))
    FLAIR_path = os.path.join('./RESECT_Source', path_dir, 'MRI', '{}-FLAIR.nii.gz'.format(path_dir))

    T1_Image = sitk.ReadImage(T1_path)
    FLAIR_Image = sitk.ReadImage(FLAIR_path)
    T1_Image = resamplevolume(T1_Image, Direction=(-1.0,0.0,0.0,0.0,-1.0,0.0,0.0,0.0,1.0))
    FLAIR_Image = resamplevolume(FLAIR_Image, Direction=(-1.0,0.0,0.0,0.0,-1.0,0.0,0.0,0.0,1.0))

    T1_volume = sitk.GetArrayFromImage(T1_Image)
    FLAIR_volume = sitk.GetArrayFromImage(FLAIR_Image)
    T1_volume = T1_volume / T1_volume.max() * 255
    FLAIR_volume = FLAIR_volume / FLAIR_volume.max() * 255
    T1_Image_new = sitk.GetImageFromArray(T1_volume)
    FLAIR_Image_new = sitk.GetImageFromArray(FLAIR_volume)
    T1_Image_new.CopyInformation(T1_Image)
    FLAIR_Image_new.CopyInformation(FLAIR_Image)

    save_dir = os.path.join('./RESECT', path_dir)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    sitk.WriteImage(T1_Image_new, os.path.join(save_dir, 'T1.nii.gz'))
    sitk.WriteImage(FLAIR_Image_new, os.path.join(save_dir, 'FLAIR.nii.gz'))

    print('{} has done...'.format(path_dir))


        