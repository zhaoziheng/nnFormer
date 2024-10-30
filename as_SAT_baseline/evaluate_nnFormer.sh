export nnUNet_raw="/mnt/hwfile/medai/zhaoziheng/SAM/nnUNet_data/nnUNet_MedNeXt_raw"
export nnUNet_preprocessed="/mnt/hwfile/medai/zhaoziheng/SAM/nnUNet_data/nnUNet_nnFormer_preprocessed"
export nnUNet_results="/mnt/hwfile/medai/zhaoziheng/SAM/nnUNet_data/nnUNet_nnFormer_results"

srun1g \
nnFormer_predict \
-i '/mnt/hwfile/medai/zhaoziheng/SAM/nnUNet_data/nnUNet_MedNeXt_raw/Task016_BTCV/imagesTs' \
-o '/mnt/hwfile/medai/zhaoziheng/SAM/nnUNet_data/nnUNet_MedNeXt_raw/Task016_BTCV/labelsPred_nnFormer' \
-chk '/mnt/hwfile/medai/zhaoziheng/SAM/nnUNet_data/nnUNet_nnFormer_results/nnFormer/3d_fullres/Task016_BTCV/nnFormerTrainerV2_nnformer_synapse__nnFormerPlansv2.1/all/model_best' \
-t 16 \
-m 3d_fullres \
-f all \
-tr nnFormerTrainerV2_nnformer_synapse \
--disable_tta

srun0g \
python /mnt/petrelfs/zhaoziheng/Knowledge-Enhanced-Medical-Segmentation/nnFormer/as_SAT_baseline/evaluate_nnFormer.py \
--dataset 'Task016_BTCV'