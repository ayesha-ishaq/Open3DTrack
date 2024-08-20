from tqdm import tqdm
import os
import json

from ultralytics import YOLOWorld
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes

def run_nuscenes(model, scenes):

    cameras = ["CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT", "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"]
    detection = {}
    for scene_id, scene_name in enumerate(tqdm(scenes)):
        scene = sequences_by_name[scene_name]
        first_token = scene['first_sample_token']
        last_token = scene['last_sample_token']
        current_token = first_token
        detection[scene_name] = {}
        frame_id = 0
        while True:
            current_sample = nusc.get('sample', current_token)
            detection[scene_name][current_token] = {}
            for cam in cameras:
                camera = nusc.get('sample_data', current_sample['data'][cam])
                img_path = os.path.join(nusc.dataroot, camera['filename'])
                results = model.predict(img_path, verbose=False)
                detection[scene_name][current_token][cam] = results[0].boxes.data.tolist()

            if current_token == last_token:
                break
            next_token = current_sample['next']
            current_token = next_token
            frame_id += 1
            
    return detection

if __name__ == "__main__":

    # Initialize a YOLO-World model
    model = YOLOWorld("yolov8l-worldv2.pt")  # or select yolov8m/l-world.pt for different sizes

    # Define custom classes
    model.set_classes(['car',
        'person',
        'bicycle',
        'truck',
        'bus',
        'motorcycle',
        'trailer'])

    dataset_dir = "/l/users/ayesha.ishaq/nuScenes"
    output_dir =  "/l/users/ayesha.ishaq/3dmotformer/"

    version_fullname = 'v1.0-trainval'
    nusc = NuScenes(version=version_fullname, dataroot=dataset_dir, verbose=True)
    sequences_by_name = {scene["name"]: scene for scene in nusc.scene}
    splits_to_scene_names = create_splits_scenes()

    train_split = 'train' 
    val_split = 'val' 
    train_scenes = splits_to_scene_names[train_split]
    val_scenes = splits_to_scene_names[val_split]
    scenes = [train_scenes, val_scenes]
    output_dirs = [(output_dir + 'training.json'), (output_dir + 'validation.json')]
    splits = ['training', 'validation']

    for scene, output in zip(scenes, output_dirs):
        results = run_nuscenes(model, scene)
        with open(output, 'w') as f:
            json.dump(results, f)