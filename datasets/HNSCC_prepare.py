import os 
import numpy as np
import cv2
import SimpleITK as sitk


def resamplevolume_P(vol, outsize, Origin=None, Direction=None):

    inputsize = vol.GetSize()
    outspacing = [0,0,0]
    outspacing[0] = inputsize[0] / outsize[0]
    outspacing[1] = inputsize[1] / outsize[1]
    outspacing[2] = inputsize[2] / outsize[2]
    
    transform = sitk.Transform()
    transform.SetIdentity()

    resampler = sitk.ResampleImageFilter()
    resampler.SetTransform(transform)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetOutputOrigin(Origin if Origin is not None else vol.GetOrigin())
    resampler.SetOutputDirection(Direction if Direction is not None else vol.GetDirection())
    resampler.SetOutputSpacing(outspacing)
    resampler.SetSize(outsize)
    newvol = resampler.Execute(vol)
    
    return newvol


for i in range(40):
    ct_path = os.path.join('./ACRIN-HNSCC-FDG-PET-CT_Source/ct_series_1channel', str(i+1))
    pt_path = os.path.join('./ACRIN-HNSCC-FDG-PET-CT_Source/pet_series_1channel', str(i+1))
    
    ct_volume = []
    pt_volume = []

    for j in range(len(os.listdir(pt_path))):
        filename = '{}.jpg'.format(str(j+1))
        ct_image = cv2.imread(os.path.join(ct_path, filename))
        pt_image = cv2.imread(os.path.join(pt_path, filename))
        
        ct_volume.append(ct_image[:,:,0])
        pt_volume.append(pt_image[:,:,0])

        if j + 1 == 263:
            break

    ct_volume = np.stack(ct_volume)
    pt_volume = np.stack(pt_volume)

    ct_volume = sitk.GetImageFromArray(ct_volume)
    pt_volume = sitk.GetImageFromArray(pt_volume)
    ct_volume = resamplevolume_P(ct_volume, [256,256,160])
    pt_volume = resamplevolume_P(pt_volume, [256,256,160])
    ct_volume.SetSpacing([1.0, 1.0, 1.0])
    pt_volume.SetSpacing([1.0, 1.0, 1.0])

    save_dir = './ACRIN-HNSCC-FDG-PET-CT/image_{}/'.format(str(i))
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    sitk.WriteImage(ct_volume, os.path.join(save_dir, 'ct.nii.gz'))
    sitk.WriteImage(pt_volume, os.path.join(save_dir, 'pet.nii.gz'))


        