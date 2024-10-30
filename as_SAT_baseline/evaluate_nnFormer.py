import json
import numpy as np
import nibabel as nb
import os
from collections import OrderedDict
import pandas as pd
from SurfaceDice import compute_surface_distances, compute_surface_dice_at_tolerance, compute_dice_coefficient
from record_results import record

join = os.path.join

def evaluate(dataset):
    
    with open(join('/mnt/hwfile/medai/zhaoziheng/SAM/nnUNet_data/nnUNet_MedNeXt_raw/{}/dataset.json'.format(dataset)), 'r') as f:
        tmp = json.load(f)
        modality = tmp["modality"]['0']
    
    save_path = '/mnt/hwfile/medai/zhaoziheng/SAM/nnUNet_data/nnUNet_MedNeXt_raw/{}/labelsPred_nnFormer'.format(dataset)
    os.makedirs(save_path, exist_ok=True)

    seg_path = join('/mnt/hwfile/medai/zhaoziheng/SAM/nnUNet_data/nnUNet_MedNeXt_raw/{}/labelsPred_nnFormer'.format(dataset))
    gt_path = join('/mnt/hwfile/medai/zhaoziheng/SAM/nnUNet_data/nnUNet_MedNeXt_raw/{}/labelsTs'.format(dataset))
    csv_path = '/mnt/hwfile/medai/zhaoziheng/SAM/nnUNet_data/nnUNet_MedNeXt_raw/{}/labelsPred_nnFormer/{}_results.csv'.format(dataset, dataset)

    filenames = np.sort(os.listdir(seg_path))
    filenames = [x for x in filenames if x.endswith('.nii.gz')]
    
    with open('/mnt/hwfile/medai/zhaoziheng/SAM/processed_files_v4/(before NC) Statistic Files/nnunet_label_mapping.json', 'r') as f:
        label_mapping = json.load(f)
    label_names = label_mapping[dataset.split('_')[-1]]

    results_of_samples = []

    for name in filenames:
        # load grond truth and segmentation
        gt_nii = nb.load(join(gt_path, name))
        case_spacing = gt_nii.header.get_zooms()
        gt_data = np.uint8(gt_nii.get_fdata())
        seg_data = np.uint8(nb.load(join(seg_path, name)).get_fdata())
        
        scores = []
        labels = []

        for organ in label_names.keys():
            ind = label_names[organ]
            
            if isinstance(ind, int):
                organ_gt, organ_seg = gt_data==ind, seg_data==ind
            else:
                organ_gt = np.zeros(gt_data.shape, dtype=np.uint8)
                organ_seg = np.zeros(seg_data.shape, dtype=np.uint8)
                for i in ind:
                    organ_gt += gt_data==i
                    organ_seg += seg_data==i
                organ_gt = organ_gt > 0
                organ_seg = organ_seg > 0

            if np.sum(organ_gt)==0 and np.sum(organ_seg)==0:
                DSC = 1
                NSD = 1
            elif np.sum(organ_gt)==0 and np.sum(organ_seg)>0:
                DSC = 0
                NSD = 0
            else:
                surface_distances = compute_surface_distances(organ_gt, organ_seg, case_spacing)
                DSC = compute_dice_coefficient(organ_gt, organ_seg)
                NSD = compute_surface_dice_at_tolerance(surface_distances, 1)
            
            print(name, organ, round(DSC,4), round(NSD,4))
            
            scores.append({'dice':round(DSC, 4), 'nsd':round(NSD, 4)})
            labels.append(organ)
        
        results_of_samples.append([dataset, modality, name, scores, labels])
        
    record(results_of_samples, csv_path)
    
    print(f'Save results to {csv_path}')
                
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Compute Dice scores')
    parser.add_argument('--dataset', type=str, default='Task016_BTCV')
    
    args = parser.parse_args()
    
    evaluate(args.dataset)
