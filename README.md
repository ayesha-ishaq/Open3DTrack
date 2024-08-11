### Acknowledgement [[ICCV2023] 3DMOTFormer: Graph Transformer for Online 3D Multi-Object Tracking](https://github.com/dsx0511/3DMOTFormer.git)

## Installation
First, clone this repository and the git submodules:
```
git clone --recurse-submodules https://github.com/ayesha-ishaq/3DLangTracker.git
```

### Conda environment
Basic installation:
```
conda create -n 3dmotformer python==3.7.13
conda activate 3dmotformer
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
pip install nuscenes-devkit matplotlib pandas motmetrics==1.1.3
pip install transformers accelerate
```

Install pytorch geometric and dependencies:
```
conda install pyg -c pyg
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-1.13.0+cu117.html
```

### Third-party dependencies
To enable using BEV IoU as matching distance for target assignment, please install the `iou3d_nms` CUDA operation from [CenterPoint](https://github.com/tianweiy/CenterPoint):

```
conda install -c conda-forge cudatoolkit-dev==11.7.0
cd CenterPoint/det3d/ops/iou3d_nms/
python setup.py install
```

To apply NMS during data pre-processing following [SimpleTrack](https://github.com/tusen-ai/SimpleTrack), please install:

```
cd SimpleTrack/
pip install -e .
```

## Data preparation

### 1. Download nuScenes
Please download nuScenes [here](https://www.nuscenes.org/download). Place nuScenes in your `$NUSCENES_DIR`.

### 2. Get detection results from an existing 3D detector
3DMOTFormer is compatible with any 3D detectors. 
You will first get the detection results as dataset to train 3DMORFormer.

#### CenterPoint (recommended)
Most existing MOT paper use [CenterPoint](https://github.com/tianweiy/CenterPoint) as public detection due to its better performance.
Following this [Github issue](https://github.com/tianweiy/CenterPoint/issues/249), you can download CenterPoint public detections that are provided by the authors:
- [With flip augmentation](https://mitprod-my.sharepoint.com/:f:/g/personal/tianweiy_mit_edu/Eip_tOTYSk5JhdVtVzlXlyABDPnGx9vsnwdo5SRK7bsh8w?e=vSdija) (68.5 NDS val performance)
- [Without flip augmentation](https://mitprod-my.sharepoint.com/:f:/g/personal/tianweiy_mit_edu/Er_nsH9Z2tRHnptBFJ_ompAByE3zu4E88xae691xyS6q_w?e=UqTmU2) (66.8 NDS val performance)

To reproduce our results reported in the paper, please use the one [with flip augmentation](https://mitprod-my.sharepoint.com/:f:/g/personal/tianweiy_mit_edu/Eip_tOTYSk5JhdVtVzlXlyABDPnGx9vsnwdo5SRK7bsh8w?e=vSdija).

#### NuScenes public detections
[Nuscenes tracking benchmark](https://www.nuscenes.org/tracking) provided several public detections:
- [MEGVII](https://www.nuscenes.org/data/detection-megvii.zip) (62.8 NDS val performance)
- [PointPillars](https://www.nuscenes.org/data/detection-pointpillars.zip) (44.8 NDS val performance)
- [Mapillary](https://www.nuscenes.org/data/detection-mapillary.zip) (36.9 NDS val performance)

#### Other detectors
If you consider using detections from another 3D detector. Please follow the instructions of their specific source code and export the results as a json file following the nuScenes output format. For example, if you want to use [BEVFusion](https://github.com/mit-han-lab/bevfusion) detections, you can follow this [Github issue](https://github.com/mit-han-lab/bevfusion/issues/233) to get the json files.

### 3. Data pre-processing
Please rename the json files with detection results for train, validation and test set as `train.json`, `val.json` and `test.json` and place them in the same folder (`$DETECTION_DIR`).
Use this [script](./generate_data_lang.py) to pre-process the detections and generate language features:
```
python generate_data_lang.py --dataset_dir=$NUSCENES_DIR --detection_dir=$DETECTION_DIR --output_dir=$PKL_DATA_DIR --apply_nms
```
This converts the json format into pkl files for all key frames and store them in the `$PKL_DATA_DIR`, which will be loaded by the dataloader during training and evaluation.

## Training and evaluation
Change the the corresponding fields to the paths to your `$NUSCENES_DIR` and `$PKL_DATA_DIR` in the [config file](./config/default.json).
To start the training, run 
```
python train.py -c config/default.json
```


## Checkpoints
Checkpoints on various language cues:

| Annotation | Weights |
|:-:|:-:|
| Instance | [link](https://mbzuaiac-my.sharepoint.com/:u:/g/personal/ayesha_ishaq_mbzuai_ac_ae/ETR0ODAL20FBuSaJCIc-tVkBz6rMXAPA2h1nJ48uueBmsA?e=tJe0tx) |
| Scene | [link](https://mbzuaiac-my.sharepoint.com/:u:/g/personal/ayesha_ishaq_mbzuai_ac_ae/ETSuCosOO8xDrHGWnkPoEOQBu1XfcuumsbOdsJluhPVj1A?e=rKHAbQ) |
| Both | [link](https://mbzuaiac-my.sharepoint.com/:u:/g/personal/ayesha_ishaq_mbzuai_ac_ae/EUFjEZoUtkRCuXXtnNO3_pQBVGhNg163tcHsjQeCxx4XBw?e=xjPASr)	|

