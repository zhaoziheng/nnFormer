import json
import os
from tqdm import tqdm
from pathlib import Path
import cv2
import numpy as np
import shutil
import random
from torchvision import transforms
from einops import repeat
import SimpleITK as sitk

def prepare_nnFormer_raw_png(
    dataset_id, 
    dataset_name, 
    nnunet_raw_data_dir='/mnt/hwfile/medai/zhaoziheng/SAM/nnUNet_data/nnUNet_raw',
    nnformer_raw_data_dir='/mnt/hwfile/medai/zhaoziheng/SAM/nnUNet_data/nnUNet_MedNeXt_raw'):    # nnFormer和MedNeXt共享raw路径，都属于nnunet_v1
        
    nnformer_raw_data_dir = os.path.join(nnformer_raw_data_dir, f'Task{dataset_id}_'+dataset_name)
    Path(os.path.join(nnformer_raw_data_dir, 'imagesTr')).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(nnformer_raw_data_dir, 'labelsTr')).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(nnformer_raw_data_dir, 'imagesTs')).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(nnformer_raw_data_dir, 'labelsTs')).mkdir(parents=True, exist_ok=True)
    
    training_samples = []
    source_imagesTr = os.path.join(nnunet_raw_data_dir, f'Dataset{dataset_id}_'+dataset_name, 'imagesTr')
    source_labelsTr = os.path.join(nnunet_raw_data_dir, f'Dataset{dataset_id}_'+dataset_name, 'labelsTr')
    for potential_img in os.listdir(source_imagesTr):
        if potential_img.endswith('0000.png'):
            
            image = cv2.imread(os.path.join(source_imagesTr, potential_img), cv2.IMREAD_GRAYSCALE)
            assert image.shape == (512, 512), f'image shape {image.shape}'
            # change 2D to 3D and save as nifiti
            image_3d = image[np.newaxis, :, :]
            image_name = potential_img.split('_0000.png')[0]    # MR_20_T2SPIR_DICOM_anon_301__s25_0000.png --> MR_20_T2SPIR_DICOM_anon_301__s25
            output_nii_path = os.path.join(nnformer_raw_data_dir, "imagesTr", image_name+'_0000.nii.gz')
            itk_img = sitk.GetImageFromArray(image_3d)
            itk_img.SetSpacing([1, 1, 999])
            sitk.WriteImage(itk_img, output_nii_path)
            
            # find corresponding mask
            assert os.path.exists(os.path.join(source_labelsTr, image_name+'.png')), f'{image_name}.png not exists'
            sc_mask = cv2.imread(os.path.join(source_labelsTr, image_name+'.png'), cv2.IMREAD_GRAYSCALE)
            assert np.where(sc_mask==0, 1, 0).sum() + np.where(sc_mask==1, 1, 0).sum() + np.where(sc_mask==2, 1, 0).sum() + np.where(sc_mask==3, 1, 0).sum() + np.where(sc_mask==4, 1, 0).sum() == 512*512, f'{np.where(sc_mask==0, 1, 0).sum()}, {np.where(sc_mask==1, 1, 0).sum()}, {np.where(sc_mask==2, 1, 0).sum()}, {np.where(sc_mask==3, 1, 0).sum()}, {np.where(sc_mask==4, 1, 0).sum()}, {sc_mask.shape}, , sc_mask max {sc_mask.max()}'
            assert sc_mask.shape == (512, 512), f'sc_mask shape {sc_mask.shape}'
            
            sc_mask_3d = sc_mask[np.newaxis, :, :]
            sc_mask_3d = sc_mask_3d.astype(np.uint32)
            output_nii_path = os.path.join(nnformer_raw_data_dir, "labelsTr", image_name+'.nii.gz')
            itk_img = sitk.GetImageFromArray(sc_mask_3d)
            itk_img.SetSpacing([1, 1, 999])
            sitk.WriteImage(itk_img, output_nii_path)
            
            training_samples.append(
                {
                    "image": os.path.join(nnformer_raw_data_dir, "imagesTr", image_name+'.nii.gz'),
                    "label": os.path.join(nnformer_raw_data_dir, "labelsTr", image_name+'.nii.gz'),
                }
            )
            
    testing_samples = []
    source_imagesTs = os.path.join(nnunet_raw_data_dir, f'Dataset{dataset_id}_'+dataset_name, 'imagesTs')
    source_labelsTs = os.path.join(nnunet_raw_data_dir, f'Dataset{dataset_id}_'+dataset_name, 'labelsTs')
    for potential_img in os.listdir(source_imagesTs):
        if potential_img.endswith('0000.png'):
            
            image = cv2.imread(os.path.join(source_imagesTs, potential_img), cv2.IMREAD_GRAYSCALE)
            assert image.shape == (512, 512), f'image shape {image.shape}'
            # change 2D to 3D and save as nifiti
            image_3d = image[np.newaxis, :, :]
            image_name = potential_img.split('_0000.png')[0]    # MR_20_T2SPIR_DICOM_anon_301__s25_0000.png --> MR_20_T2SPIR_DICOM_anon_301__s25
            output_nii_path = os.path.join(nnformer_raw_data_dir, "imagesTs", image_name+'_0000.nii.gz')
            itk_img = sitk.GetImageFromArray(image_3d)
            itk_img.SetSpacing([1, 1, 999])
            sitk.WriteImage(itk_img, output_nii_path)
            
            # find corresponding mask
            assert os.path.exists(os.path.join(source_labelsTs, image_name+'.png')), f'{image_name}.png not exists'
            sc_mask = cv2.imread(os.path.join(source_labelsTs, image_name+'.png'), cv2.IMREAD_GRAYSCALE)
            assert np.where(sc_mask==0, 1, 0).sum() + np.where(sc_mask==1, 1, 0).sum() + np.where(sc_mask==2, 1, 0).sum() + np.where(sc_mask==3, 1, 0).sum() + np.where(sc_mask==4, 1, 0).sum() == 512*512, f'{np.where(sc_mask==0, 1, 0).sum()}, {np.where(sc_mask==1, 1, 0).sum()}, {np.where(sc_mask==2, 1, 0).sum()}, {np.where(sc_mask==3, 1, 0).sum()}, {np.where(sc_mask==4, 1, 0).sum()}, {sc_mask.shape}, , sc_mask max {sc_mask.max()}'
            assert sc_mask.shape == (512, 512), f'sc_mask shape {sc_mask.shape}'
            
            sc_mask_3d = sc_mask[np.newaxis, :, :]
            sc_mask_3d = sc_mask_3d.astype(np.uint32)
            output_nii_path = os.path.join(nnformer_raw_data_dir, "labelsTs", image_name+'.nii.gz')
            itk_img = sitk.GetImageFromArray(sc_mask_3d)
            itk_img.SetSpacing([1, 1, 999])
            sitk.WriteImage(itk_img, output_nii_path)
            
            testing_samples.append(os.path.join(nnformer_raw_data_dir, "imagesTr", image_name+'.nii.gz'))
    
    with open(os.path.join(nnunet_raw_data_dir, f'Dataset{dataset_id}_'+dataset_name, 'dataset.json'), 'r') as f:
        source_dataset_json = json.load(f)       
    
    source_dataset_json['training'] = training_samples
    source_dataset_json['test'] = testing_samples
    source_dataset_json['numTest'] = len(testing_samples)
    source_dataset_json['modality'] = source_dataset_json['channel_names']
    del source_dataset_json['channel_names']
    
    # 3. change labels from {"background":"0", ...} to {"0":"background", ...}
    new_labels = {v:k for k,v in source_dataset_json['labels'].items()}
    source_dataset_json['labels'] = new_labels

    with open(f'{nnformer_raw_data_dir}/dataset.json', 'w') as f:
        json.dump(source_dataset_json, f, indent=4)
    
    print(f'Create Dataset{dataset_id}_{dataset_name} under nnUNet_raw, {len(training_samples)} training samples and {len(testing_samples)} testing samples')
        
if __name__ == '__main__':
    
    # 2D nnunet v2 raw_data --> 2D nnunet v1 raw_data
    
    prepare_nnFormer_raw_png(dataset_id=992, dataset_name='CHAOSMR_T2SPIR_PNG')