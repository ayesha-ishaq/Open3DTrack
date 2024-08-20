#!/bin/bash
#SBATCH --job-name=3dmot             # Job name
#SBATCH --output=output.%A_%a.txt   # Standard output and error log
#SBATCH --error=error.%A_%a.txt
#SBATCH -w ws-l5-012
#SBATCH -N1
#SBATCH -n24
#SBATCH -t 24:00:00

source activate 3dmotformer
export PYTHONPATH="${PYTHONPATH}:/home/ayesha.ishaq/Desktop/3DMOTFormer/nuscenes-devkit/python-sdk"
# python generate_data_yoloworld.py --dataset_dir /l/users/ayesha.ishaq/nuScenes --detection_dir /home/ayesha.ishaq/Desktop/3DMOTFormer/detections --output_dir /l/users/ayesha.ishaq/3dmotformer/output_pkl_label_features --yoloworld_dir /l/users/ayesha.ishaq/3dmotformer --apply_nms
# python data_preprocess.py --output_dir /l/users/ayesha.ishaq/3dmotformer/output_pkl_label_features --save_dir /l/users/ayesha.ishaq/3dmotformer/output_pkl_split
# python generate_data_openscene.py --dataset_dir /l/users/ayesha.ishaq/nuScenes --detection_dir /home/ayesha.ishaq/Desktop/3DMOTFormer/detections --output_dir /l/users/ayesha.ishaq/3dmotformer/output_pkl_label_features --openscene_dir /l/users/ayesha.ishaq/openscene/ --apply_nms
python train.py -c config/default.json -r "/home/ayesha.ishaq/Desktop/3DMOTFormer/workspace/default/0815_125855/models/checkpoint-epoch12.pth" --eval_only -o "/home/ayesha.ishaq/Desktop/3DMOTFormer/open_eval/"  
# "/home/ayesha.ishaq/Desktop/3DMOTFormer/workspace/default/0418_163414_boxes/models/checkpoint-epoch12.pth" path to lang boxes only
# /home/ayesha.ishaq/Desktop/3DMOTFormer/workspace/default/0422_024224_scene_boxes/models/checkpoint-epoch12.pth path to lang scene and boxes
# "/home/ayesha.ishaq/Desktop/3DMOTFormer/workspace/default/0422_172500_scene/models/checkpoint-epoch12.pth" path to lang scene only

# 0707_145119 exp I

# source activate sam3d
# python yoloworld.py