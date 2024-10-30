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

COLORS = [
    [0, 0, 0],
    [0, 0, 255],     # 红色
    [0, 255, 0],     # 绿色
    [255, 0, 0],     # 蓝色
    [0, 255, 255],   # 黄色
    [255, 0, 255],   # 紫色
    [255, 255, 0],   # 青色
    [128, 0, 0],     # 暗红色
    [0, 128, 0],     # 暗绿色
    [0, 0, 128],     # 暗蓝色
    [128, 128, 0],   # 橄榄色
    [128, 0, 128],   # 深紫色
    [0, 128, 128],   # 青绿
    [192, 192, 192], # 银色
    [128, 128, 128], # 灰色
    [0, 0, 0],       # 黑色
    [255, 255, 255], # 白色
    [255, 165, 0],   # 橙色
    [75, 0, 130],    # 靛青色
    [238, 130, 238], # 粉紫色
    [173, 216, 230], # 浅蓝色
]

def generate_dataset_json(
    raw_data_dir,   
    modality,
    labels,
    training_samples,
    test_samples,
    name,
    ):
    """
    {
    "labels": {
        "0": "background",
        "1": "spleen",
        ...
    },
    "numTraining": x,
    "file_ending": ".nii.gz",
    "name": "BTCV",
    "reference": "none",
    "description": "BTCV",
    "overwrite_image_reader_writer": "NibabelIOWithReorient",
    "training": [
        {
            "image": "/mnt/hwfile/medai/zhaoziheng/SAM/nnUNet_data/nnUNet_raw/Dataset016_BTCV/imagesTr/img0032.nii.gz",
            "label": "/mnt/hwfile/medai/zhaoziheng/SAM/nnUNet_data/nnUNet_raw/Dataset016_BTCV/labelsTr/img0032.nii.gz"
        },
        ...
    ],
    "test": [
        "/mnt/hwfile/medai/zhaoziheng/SAM/nnUNet_data/nnUNet_raw/Dataset016_BTCV/imagesTs/img0034.nii.gz",
        ...
    ],
    "numTest": x,
    "modality": {
        "0": "CT"
    }
    }
    """
    
    data_dict = {}
    data_dict['labels'] = labels
    data_dict['numTraining'] = len(training_samples)
    data_dict['file_ending'] = ".nii.gz"
    data_dict['name'] = name
    data_dict['reference'] = 'none'
    data_dict['description'] = name
    data_dict['overwrite_image_reader_writer'] = "NibabelIOWithReorient"
    data_dict['training'] = training_samples
    data_dict['test'] = test_samples
    data_dict['numTest'] = len(test_samples)
    data_dict['modality'] = modality
    
    with open(os.path.join(raw_data_dir, 'dataset.json'), 'w') as f:
        json.dump(data_dict, f, indent=4)

def resize_with_padding(slice, target_size=512, nearest_mode=True):
    # 获取原始数组的高度和宽度
    h, w = slice.shape
    
    # 计算需要填充的尺寸
    if h > w:
        pad_width = (h - w) // 2
        padding = ((0, 0), (pad_width, h - w - pad_width))
    else:
        pad_height = (w - h) // 2
        padding = ((pad_height, w - h - pad_height), (0, 0))
    
    # 使用 np.pad 进行填充
    padded = np.pad(slice, padding, mode='constant', constant_values=0)
    
    # 确保填充后的尺寸是方形
    assert padded.shape[0] == padded.shape[1], "Padding did not result in a square matrix."
    
    # 使用 OpenCV 进行缩放（插值方式可根据需求更改）
    if nearest_mode:
        resized = cv2.resize(padded, (target_size, target_size), interpolation=cv2.INTER_NEAREST)
    else:
        resized = cv2.resize(padded, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
    
    return resized

def prepare_nnFormer_raw_png(
    json_file, 
    filter_ls_json,
    mask2label, 
    dataset_id, 
    dataset_name, 
    raw_data_dir='/mnt/hwfile/medai/zhaoziheng/SAM/nnUNet_data/nnUNet_MedNeXt_raw'):    # nnFormer和MedNeXt共享raw路径，都属于nnunet_v1
    
    if filter_ls_json:
        with open(filter_ls_json, 'r') as f:
            filter_ls = json.load(f)

    modality = {0:"MR"}
    file_ending = '.png'

    with open(json_file, 'r') as f:
        data = json.load(f)
        
    raw_data_dir = os.path.join(raw_data_dir, f'Task{dataset_id}_'+dataset_name)
    Path(os.path.join(raw_data_dir, 'imagesTr')).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(raw_data_dir, 'labelsTr')).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(raw_data_dir, 'visualizationTr')).mkdir(parents=True, exist_ok=True)
    
    labels = {
        "0":"background"
    }
    cursor = 1
    for mask_path_key, label_name in mask2label.items():
        labels[str(cursor)] = label_name
        cursor += 1
    
    num_train = 0
    training_samples = []
    
    for datum in tqdm(data):
        
        # {
        #     "prompt": [
        #         "T2 SPIR Abdomen MRI, fat low signal, muscle low signal, water high signal, fat dark, water bright, Upper Abdominal Region, liver, right kidney, left kidney, spleen, kidney"
        #     ],
        #     "modality": "T1 in-phase Abdomen MRI",
        #     "region": "Upper Abdominal Region",
        #     "label": "liver, right kidney, left kidney, spleen, kidney",
        #     "mask_path": "/mnt/petrelfs/wuhaoning/MedicalGen/MRI_diffusion/synthetic_mask/0_mask.png",
        #     "mask_0_path": "/mnt/petrelfs/wuhaoning/MedicalGen/MRI_diffusion/synthetic_mask/0_mask_0.png",    # liver
        #     "mask_1_path": "/mnt/petrelfs/wuhaoning/MedicalGen/MRI_diffusion/synthetic_mask/0_mask_1.png",    # right kidney
        #     "mask_2_path": "/mnt/petrelfs/wuhaoning/MedicalGen/MRI_diffusion/synthetic_mask/0_mask_2.png",    # left kidney
        #     "mask_3_path": "/mnt/petrelfs/wuhaoning/MedicalGen/MRI_diffusion/synthetic_mask/0_mask_3.png",    # spleen
        #     "image_path": "/mnt/petrelfs/wuhaoning/MedicalGen/MRI_diffusion/synthetic_T2-SPIR/0_T2-SPIR Abdomen MRI_0.png",
        #     "combined_path": "/mnt/petrelfs/wuhaoning/MedicalGen/MRI_diffusion/synthetic_T2-SPIR_combined/0_T2-SPIR Abdomen MRI_0_combined.png"
        # },
        
        # load image
        image_path = datum['image_path']
        if filter_ls_json:
            if image_path not in filter_ls:
                continue
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        height, width = image.shape
        
        # merge masks into a sc mask
        color_mask = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
        sc_mask = []
        for mask_path_key, label_name in mask2label.items():
            source_path = datum[mask_path_key]
            tmp = cv2.imread(source_path, cv2.IMREAD_GRAYSCALE)
            tmp = np.where(tmp>0, 1, 0)
            assert tmp.max() <= 1 and tmp.min() == 0, f"{source_path} out of boundary: max={tmp.max()}, min={tmp.min()}"
            sc_mask.append(labels[label_name] * tmp)
            color_mask[tmp == 1] = COLORS[labels[label_name]]
        sc_mask = np.stack(sc_mask, axis=0) # 4, 512, 512
        sc_mask = np.max(sc_mask, axis=0)
            
        # check format
        assert np.where(sc_mask==0, 1, 0).sum() + np.where(sc_mask==1, 1, 0).sum() + np.where(sc_mask==2, 1, 0).sum() + np.where(sc_mask==3, 1, 0).sum() + np.where(sc_mask==4, 1, 0).sum() == 512*512, f'{np.where(sc_mask==0, 1, 0).sum()}, {np.where(sc_mask==1, 1, 0).sum()}, {np.where(sc_mask==2, 1, 0).sum()}, {np.where(sc_mask==3, 1, 0).sum()}, {np.where(sc_mask==4, 1, 0).sum()}, {sc_mask.shape}, , sc_mask max {sc_mask.max()}'
        assert image.shape == sc_mask.shape == (512, 512), f'image shape {image.shape}, sc_mask shape {sc_mask.shape}'

        # change 2D to 3D and save as nifiti
        image_3d = image[np.newaxis, :, :]
        image_name = image_path.split('/')[-1].split('.')[0]
        output_nii_path = os.path.join(raw_data_dir, "imagesTr", image_name+'_0000.nii.gz')
        itk_img = sitk.GetImageFromArray(image_3d)
        itk_img.SetSpacing([1, 1, 999])
        sitk.WriteImage(itk_img, output_nii_path)
        
        sc_mask_3d = sc_mask[np.newaxis, :, :]
        sc_mask_3d = sc_mask_3d.astype(np.uint32)
        output_nii_path = os.path.join(raw_data_dir, "labelsTr", image_name+'.nii.gz')
        itk_img = sitk.GetImageFromArray(sc_mask_3d)
        itk_img.SetSpacing([1, 1, 999])
        sitk.WriteImage(itk_img, output_nii_path)
        
        training_samples.append({
            "image": os.path.join(raw_data_dir, "imagesTr", image_name+'.nii.gz'),
            "label": os.path.join(raw_data_dir, "labelsTr", image_name+'.nii.gz')
        })
        
        # visualization
        color_image = repeat(image, 'h w -> h w c', c=3)
        color_image = (color_image - color_image.min()) / (color_image.max() - color_image.min()) * 255.0
        color_image = color_image * 0.7 + color_mask * 0.3
        cv2.imwrite(os.path.join(raw_data_dir, "visualizationTr", image_name+'.png'), color_image)
        
        num_train += 1
        
    generate_dataset_json(
        raw_data_dir, 
        modality=modality,
        labels=labels,
        training_samples=training_samples, 
        name=dataset_name, 
        test_samples=[],
        )
    
    print(f'Create Dataset{dataset_id}_{dataset_name} under nnUNet_raw, {num_train} training samples')
    
def merg_json(json_ls, ratio_ls, save_path):
    data = []
    for json_file, ratio in zip(json_ls, ratio_ls):
        sample_num = 0
        
        with open(json_file, 'r') as f:
            tmp = json.load(f)
            
        for datum in tmp:
            if random.random() < ratio:
                data.append(datum)
                sample_num += 1
                
        print(f'{json_file} sample num: {sample_num}')
            
    with open(save_path, 'w') as f:
        json.dump(data, f, indent=4)
        
if __name__ == '__main__':
    
    # json file --> nnunet v1 raw data
    
    mask2label_chaos = {
        'mask_0_path':'liver',
        'mask_1_path':'right kidney',
        'mask_2_path':'left kidney',
        'mask_3_path':'spleen',
    }
    
    mask2label_prostate = {
        'mask_path':'prostate'
    }
    
    mask2label_liver = {
        'mask_path':'liver'
    }
    
    mask2label_liver_tumor = {
        'mask0_path':'liver',
        'mask1_path':'liver tumor'
    }
    
    mask2label_pancreas = {
        'mask_path':'pancreas'
    }
    
    # prepare_nnUNet_raw_png(
    #     json_file='/mnt/hwfile/medai/zhaoziheng/SAM/MRDiffusion/synthetic_CHAOSonly_without_nomask/json_files/synthetic_T2-SPIR.json',
    #     mask2label=mask2label, 
    #     dataset_id=998, 
    #     dataset_name='CHAOSMR_T2SPIR_whn0822'
    #     )
    
    
    
    