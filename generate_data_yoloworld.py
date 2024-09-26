import pickle
import os
import cv2
import json
import math
from tqdm import tqdm
import argparse
import numpy as np
from PIL import Image
from pathlib import Path
from collections import OrderedDict
from pyquaternion import Quaternion

from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes
from nuscenes.utils.data_classes import Box
from shapely.geometry import box as B
from shapely.geometry import MultiPoint, LineString
from nuscenes.utils.geometry_utils import view_points
from nuscenes.eval.common.loaders import load_prediction
from nuscenes.eval.detection.data_classes import DetectionBox
from nuscenes.eval.common.utils import quaternion_yaw
from nuscenes.eval.tracking.utils import category_to_tracking_name

from utils.data_util import NuScenesClasses, NuScenesClassesBase, category_to_tracking_name_base


def post_process_coords(corner_coords, imsize=(1600, 900)):
    """
    Get the intersection of the convex hull of the reprojected bbox corners and the image canvas, return None if no
    intersection.
    :param corner_coords: Corner coordinates of reprojected bounding box.
    :param imsize: Size of the image canvas.
    :return: Intersection of the convex hull of the 2D box corners and the image canvas.
    """
    polygon_from_2d_box = MultiPoint(corner_coords).convex_hull
    img_canvas = B(0, 0, imsize[0], imsize[1])

    if polygon_from_2d_box.intersects(img_canvas):
        img_intersection = polygon_from_2d_box.intersection(img_canvas)
        if isinstance(img_intersection, LineString):
          return None
        intersection_coords = np.array([coord for coord in img_intersection.exterior.coords])

        min_x = int(min(intersection_coords[:, 0]))
        min_y = int(min(intersection_coords[:, 1]))
        max_x = int(max(intersection_coords[:, 0]))
        max_y = int(max(intersection_coords[:, 1]))

        return min_x, min_y, max_x, max_y
    else:
        return None

def simpletrack_nms(frame_det_data, iou_threshold=0.1):
    from SimpleTrack.data_loader.nuscenes_loader import nu_array2mot_bbox
    from mot_3d.preprocessing import nms

    boxes = np.concatenate([frame_det_data['translation'],
                            frame_det_data['size'],
                            frame_det_data['rotation'],
                            np.expand_dims(frame_det_data['score'], axis=1)], 
                            axis=1)
    classes = frame_det_data['class']
    boxes_mot = [nu_array2mot_bbox(b) for b in boxes]

    index, _ = nms(boxes_mot, classes, iou_threshold)

    frame_det_data['translation'] = frame_det_data['translation'][index]
    frame_det_data['size'] = frame_det_data['size'][index]
    frame_det_data['yaw'] = frame_det_data['yaw'][index]
    frame_det_data['rotation'] = frame_det_data['rotation'][index]
    frame_det_data['velocity'] = frame_det_data['velocity'][index]
    frame_det_data['class'] = frame_det_data['class'][index]
    frame_det_data['score'] = frame_det_data['score'][index]

    return frame_det_data


def compute_iou(box1, box2):
    """
    Compute the Intersection over Union (IoU) of two boxes.
    
    Args:
    box1: A list of 4 numbers representing the coordinates of the first box [x1, y1, x2, y2].
    box2: A list of 4 numbers representing the coordinates of the second box [x1, y1, x2, y2].
    
    Returns:
    iou: IoU value (float) between the two boxes.
    """
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])

    # Compute the area of intersection rectangle
    inter_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)
    
    # Compute the area of both the prediction and ground-truth rectangles
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    # Compute the intersection over union
    iou = inter_area / float(box1_area + box2_area - inter_area)
    
    return iou

def compute_center(box):
    """
    Compute the center of a box.
    
    Args:
    box: A list of 4 numbers representing the coordinates of the box [x1, y1, x2, y2].
    
    Returns:
    center: A tuple (x_center, y_center) representing the center of the box.
    """
    x_center = (box[0] + box[2]) / 2
    y_center = (box[1] + box[3]) / 2
    return (x_center, y_center)

def compute_distance(box1, box2):
    """
    Compute the Euclidean distance between the centers of two boxes.
    
    Args:
    box1: A list of 4 numbers representing the coordinates of the first box [x1, y1, x2, y2].
    box2: A list of 4 numbers representing the coordinates of the second box [x1, y1, x2, y2].
    
    Returns:
    distance: Euclidean distance between the centers of the two boxes.
    """
    center1 = compute_center(box1)
    center2 = compute_center(box2)
    distance = math.sqrt((center1[0] - center2[0]) ** 2 + (center1[1] - center2[1]) ** 2)
    return distance

def calculate_weight(det_box, image_height=900):
    # Calculate base depth estimate (inverse of size)
    box_width = det_box[2] - det_box[0]
    box_height = det_box[3] - det_box[1]
    box_size = (box_width * box_height)/ (1600 * 900)
    base_depth_estimate = 1 / box_size if box_size > 0 else float('inf')
    
    y_center =  compute_center(det_box)[1]
    # Apply perspective correction
    perspective_factor = 1 - 0.2*(y_center / image_height)
    adjusted_depth = base_depth_estimate * perspective_factor
    
    # Optionally adjust for aspect ratio if needed
    aspect_ratio = box_width / box_height
    if aspect_ratio > 2.5:  # e.g., large aspect ratio may indicate a distant large object
        adjusted_depth = adjusted_depth * (1 / aspect_ratio)
    
    # Calculate weight (larger weight for closer objects)
    weight = adjusted_depth if adjusted_depth > 0 else float('inf')
    return weight

def find_max_iou_or_closest(projected_box, detection_boxes):
    """
    Compute the IoU between the projected box and all detection boxes, and find the one with the maximum IoU.
    If no overlap, return None.
    
    Args:
    projected_box: A list of 4 numbers representing the coordinates of the projected box [x1, y1, x2, y2].
    detection_boxes: A list of lists, where each list contains 4 numbers representing the coordinates of a detection box [x1, y1, x2, y2].
    
    Returns:
    best_box: The detection box with the maximum IoU, or the closest box if no overlap.
    best_value: The maximum IoU value or the minimum distance if no overlap.
    """
    if not detection_boxes:  # If there are no detection boxes
        return None, None, None

    max_iou = 0
    best_box_class = None
    score = None
    dist_weight = None

    low_iou_boxes = []
    for i, det_box in enumerate(detection_boxes):
        if det_box[-2] >= 0.02:
            iou = compute_iou(projected_box, det_box[:4])
            if iou > max_iou:
                max_iou = iou
                dist_weight = calculate_weight(det_box[:4])
                best_box_class = det_box[-1]
                score = det_box[-2]
        else:
            low_iou_boxes.append(i)

    if max_iou > 0:
        return best_box_class, score, dist_weight

    else:
        for index in low_iou_boxes:
            iou = compute_iou(projected_box, detection_boxes[i][:4])
            if iou > max_iou:
                max_iou = iou
                dist_weight = calculate_weight(det_box[:4])
                best_box_class = detection_boxes[i][-1]
                score = detection_boxes[i][-2]
    
    return best_box_class, score, dist_weight


def base_class_2d(dets, scenario='rare'):
    for scene in dets:
        for frame in dets[scene]:
            for cam in dets[scene][frame]:
                base_dets = []
                for det in dets[scene][frame][cam]:
                    if det[-1] in NuScenesClassesBase[scenario].values():
                        base_dets.append(det)
                dets[scene][frame][cam] = base_dets
    return dets

def write_data_per_scene(scene, output):
    print('Writing data into pkl files...')
    for i, frame in enumerate(scene):
        if i < 1:
            scene_id = frame['scene_id']
            scene_output = output / f'{scene_id:04d}'
            scene_output.mkdir(parents=True, exist_ok=True)
        
        frame_id = frame['frame_id']
        filename = scene_output / f'{frame_id:03d}.pkl'

        content = {'dets': frame['detections'],
                   'gts': frame['ground_truths'],
                   'num_dets': frame['num_dets'],
                   'num_gts': frame['num_gts'],
                   'ego_translation': frame['ego_translation'],
                   'timestamp': frame['timestamp'],
                   'token': frame['sample_token']
                  }

        with open(filename, 'wb') as f:
            pickle.dump(content, f)

def generate_nusc_seq_data(nusc, det_boxes, scenes, sequences_by_name, output, split, yoloworld_path, scenario='rare', apply_nms=False):

    print('Generating detection and ground truth sequences...')
    result = []

    cameras = ["CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT", "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"]

    with open(yoloworld_path, 'r') as f:
        dets = json.load(f)

    if split == 'training':
        dets = base_class_2d(dets, scenario)

    for scene_id, scene_name in enumerate(tqdm(scenes)):
        if scene_id >=0:
            scene = sequences_by_name[scene_name]
            first_token = scene['first_sample_token']
            last_token = scene['last_sample_token']
            current_token = first_token
            scene_result = []
            tracking_id_set = set()
    
            frame_id = 0
            while True:
                current_sample = nusc.get('sample', current_token)
                
                # Get ego pose data
                lidar_top_data = nusc.get('sample_data', current_sample['data']['LIDAR_TOP'])
                ego_pose = nusc.get('ego_pose', lidar_top_data['ego_pose_token'])
                ego_trans = np.array(ego_pose['translation'], dtype=np.float32)
                ego_timestamp = np.array(ego_pose['timestamp'], dtype=np.int)
                
                ## Process and concat detections for every frame
                frame_dets = det_boxes[current_token]
                det_trans = []
                det_size = []
                det_yaw = []
                det_rot = []
                det_velo = []
                det_class = []
                det_score = []
    
                for det in frame_dets:
                    det_dict = det.serialize()
                    if split == 'training':
                        if det_dict['detection_name'] in NuScenesClassesBase[scenario].keys():
                            det_trans.append(det_dict['translation'])
                            det_size.append(det_dict['size'])
                            det_yaw.append([quaternion_yaw(Quaternion(det_dict['rotation']))])
                            det_rot.append(det_dict['rotation'])
                            det_velo.append(det_dict['velocity'])
                            det_class.append(NuScenesClasses[det_dict['detection_name']])
                            det_score.append(det_dict['detection_score'])
                    else:
                        if det_dict['detection_name'] in NuScenesClasses.keys():
                            det_trans.append(det_dict['translation'])
                            det_size.append(det_dict['size'])
                            det_yaw.append([quaternion_yaw(Quaternion(det_dict['rotation']))])
                            det_rot.append(det_dict['rotation'])
                            det_velo.append(det_dict['velocity'])
                            det_class.append(NuScenesClasses[det_dict['detection_name']])
                            det_score.append(det_dict['detection_score'])
    
                frame_dets_dict = {
                    'translation': np.array(det_trans, dtype=np.float32), # [N, 3]
                    'size': np.array(det_size, dtype=np.float32), # [N, 3]
                    'yaw': np.array(det_yaw, dtype=np.float32), # [N, 1]
                    'rotation': np.array(det_rot, dtype=np.float32), # [N, 4]
                    'velocity': np.array(det_velo, dtype=np.float32), # [N, 2]
                    'class': np.array(det_class, dtype=np.int32), # [N]
                    'score': np.array(det_score, dtype=np.float32), # [N]
                }
    
                if apply_nms and len(frame_dets_dict['translation'])>1:
                    frame_dets_dict = simpletrack_nms(frame_dets_dict, iou_threshold=0.1)
                    
                # get 2D bounding box for nms detections only

                yolo_class = np.zeros(len(frame_dets_dict['translation']))
                yolo_score = np.zeros(len(frame_dets_dict['translation']))
                distance_weights = np.zeros(len(frame_dets_dict['translation']))
            
                for nms_dets_index in range(len(frame_dets_dict['translation'])):
                    proj_flag = False
                    for cam in cameras:
                        camera = nusc.get('sample_data', current_sample['data'][cam])
                        camera_transform = nusc.get('calibrated_sensor', camera['calibrated_sensor_token'])
                        ego_pose_cam = nusc.get('ego_pose', camera['ego_pose_token'])
                        box = Box(frame_dets_dict['translation'][nms_dets_index],
                                frame_dets_dict['size'][nms_dets_index], 
                                Quaternion(frame_dets_dict['rotation'][nms_dets_index]))
                        # Move them to the ego-pose frame.
                        box.translate(-np.array(ego_pose_cam['translation']))
                        box.rotate(Quaternion(ego_pose_cam['rotation']).inverse)

                        # Move them to the calibrated sensor frame.
                        box.translate(-np.array(camera_transform['translation']))
                        box.rotate(Quaternion(camera_transform['rotation']).inverse)

                        # Filter out the corners that are not in front of the calibrated sensor.
                        corners_3d = box.corners()
                        in_front = np.argwhere(corners_3d[2, :] > 0).flatten()
                        corners_3d = corners_3d[:, in_front]

                        # Project 3d box to 2d.
                        corner_coords = view_points(corners_3d, np.array(camera_transform['camera_intrinsic'], dtype=np.float32),
                        True).T[:, :2].tolist()

                        final_coords = post_process_coords(corner_coords)
                        
                        # Skip if the convex hull of the re-projected corners does not intersect the image canvas.
                        if final_coords is None:
                            continue
                        else:
                            class_name, class_score, dist_weight = find_max_iou_or_closest(final_coords, dets[scene_name][current_token][cam])
                            proj_flag = True
                            break
                
                    if not proj_flag or class_name==None:
                        class_name = 7
                        class_score = 0.01
                        dist_weight = 0.0
            
                    yolo_class[nms_dets_index] = int(class_name)
                    yolo_score[nms_dets_index] = class_score
                    distance_weights[nms_dets_index] = dist_weight

                frame_dets_dict['yolo_class'] = yolo_class.astype(np.int32) # [N]
                frame_dets_dict['yolo_score'] = yolo_score.astype(np.float32) # [N]
                frame_dets_dict['distance_weights'] = distance_weights.astype(np.float32) # [N]
                
                ## Process and concat ground truths for every frame
                frame_ann_tokens = current_sample['anns']
                gt_trans = []
                gt_size = []
                gt_yaw = []
                gt_rot = []
                gt_class = []
                gt_track_token = []
    
                gt_next_exist = []
                gt_next_trans = []
                gt_next_size = []
                gt_next_yaw = []
    
                for ann_token in frame_ann_tokens:
                    ann = nusc.get('sample_annotation', ann_token)
                    if split == 'training':
                        tracking_name = category_to_tracking_name_base(scenario, ann['category_name'])
                    else:
                        tracking_name = category_to_tracking_name(ann['category_name'])
                    if tracking_name is not None:
                        instance_token = ann['instance_token']
                        tracking_id_set.add(instance_token)
    
                        gt_trans.append(ann['translation'])
                        gt_size.append(ann['size'])
                        gt_yaw.append([quaternion_yaw(Quaternion(ann['rotation']))])
                        gt_rot.append(ann['rotation'])
                        gt_class.append(NuScenesClasses[tracking_name])
                        gt_track_token.append(instance_token)
    
                        next_ann_token = ann['next']
                        if next_ann_token == "":
                            gt_next_exist.append(False)
                            gt_next_trans.append([0.0, 0.0, 0.0])
                            gt_next_size.append([0.0, 0.0, 0.0])
                            gt_next_yaw.append([0.0])
                        else:
                            gt_next_exist.append(True)
                            next_ann = nusc.get('sample_annotation', next_ann_token)
                            gt_next_trans.append(next_ann['translation'])
                            gt_next_size.append(next_ann['size'])
                            gt_next_yaw.append([quaternion_yaw(Quaternion(next_ann['rotation']))])
    
                frame_anns_dict = {
                    'translation': np.array(gt_trans, dtype=np.float32), # [M, 3]
                    'size': np.array(gt_size, dtype=np.float32), # [M, 3]
                    'yaw': np.array(gt_yaw, dtype=np.float32), # [M, 1]
                    'rotation': np.array(gt_rot, dtype=np.float32), # [M, 4]
                    'class': np.array(gt_class, dtype=np.int32), # [M]
                    'tracking_id': gt_track_token, # [M]
                    'next_exist': np.array(gt_next_exist, dtype=np.bool), # [M]
                    'next_translation': np.array(gt_next_trans, dtype=np.float32), # [M, 3]
                    'next_size': np.array(gt_next_size, dtype=np.float32), # [M, 3]
                    'next_yaw': np.array(gt_next_yaw, dtype=np.float32), # [M, 1]
                }
    
                frame_result = {'detections': frame_dets_dict,
                                'ground_truths': frame_anns_dict,
                                'num_dets': len(det_trans), # int: N
                                'num_gts': len(gt_trans), # int: M
                                'scene_id': scene_id,
                                'frame_id': frame_id,
                                'ego_translation': ego_trans,
                                'timestamp': ego_timestamp,
                                'sample_token': current_token
                                }
                scene_result.append(frame_result)
    
                if current_token == last_token:
                    break
    
                next_token = current_sample['next']
                current_token = next_token
                frame_id += 1
    
            assert len(scene_result) == scene['nbr_samples']
            
            ## Convert instance token to tacking id for the whole scene
            tracking_token_to_id = {}
            for i, tracking_id in enumerate(tracking_id_set):
                tracking_token_to_id.update({tracking_id: i})
            
            for frame_result in scene_result:
                for i, tracking_token in enumerate(frame_result['ground_truths']['tracking_id']):
                    tracking_id = tracking_token_to_id[tracking_token]
                    frame_result['ground_truths']['tracking_id'][i] = tracking_id
                frame_result['ground_truths']['tracking_id'] = \
                    np.array(frame_result['ground_truths']['tracking_id'], dtype=np.int32)
    
            write_data_per_scene(scene_result, output)
            result.append(scene_result)
        
    print('Done generating.')
    print('======')
    
    return result

def generate_nusc_data(version, dataset_dir, detection_dir, output_dir, yoloworld, scenario='rare', apply_nms=False):

    dataset_dir = dataset_dir 
    train_result_file = detection_dir / "train.json"
    val_result_file = detection_dir / "val.json"
    test_result_file = detection_dir / "test.json"

    version_fullname = version
    if version == "v1.0":
        version_fullname += '-trainval'
    nusc = NuScenes(version=version_fullname, dataroot=dataset_dir, verbose=True)
    sequences_by_name = {scene["name"]: scene for scene in nusc.scene}
    splits_to_scene_names = create_splits_scenes()

    train_split = 'train' if version == "v1.0" else 'mini_train'
    val_split = 'val' if version == "v1.0" else 'mini_val'
    test_split = 'test'
    train_scenes = splits_to_scene_names[train_split]
    val_scenes = splits_to_scene_names[val_split]

    result_files = [train_result_file, val_result_file]
    scenes = [train_scenes, val_scenes]
    output_dirs = [output_dir / 'training', output_dir / 'validation']
    yoloworld_dirs = [yoloworld / 'training.json', yoloworld / 'validation.json']
    splits = ['training', 'validation']


    # Train and validation split
    for result_file, scene, output, split, yolo_det in zip(result_files, scenes, output_dirs, splits, yoloworld_dirs):
        output.mkdir(parents=True, exist_ok=True)
        print('Loading Nusences 3d detctions...')
        det_boxes, _ = load_prediction(result_file, 10000, DetectionBox, verbose=True)
        print('======')

        data = generate_nusc_seq_data(nusc, det_boxes, scene, sequences_by_name, output, split, yolo_det, scenario, apply_nms)

    


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='Nuscenes data preprocessing')
    args.add_argument('--dataset_dir', default=None, type=str,
                      help='Directory where nuScenes dataset is stored')
    args.add_argument('--version', default="v1.0", type=str,
                      help='Version of nuScenes dataset')
    args.add_argument('--detection_dir', default=None, type=str,
                      help='Directory where detection results from 3D detector are stored')
    args.add_argument('--output_dir', default=None, type=str,
                      help='Directory where preprocessed pickle files will be stored')
    args.add_argument('--yoloworld_dir', default=None, type=str,
                      help='Directory where yoloworld detection results on images are stored')
    args.add_argument('--data_split_scenario', default="rare", type=str,
                        help='Data split scenario: diverse, urban, rare')
    args.add_argument('--apply_nms', action='store_true',
                      help='Whether to apply a Non-Maximum Suppression')
    args = args.parse_args()

    generate_nusc_data(version=args.version,
                       dataset_dir=Path(args.dataset_dir),
                       detection_dir=Path(args.detection_dir),
                       output_dir=Path(args.output_dir),
                       yoloworld=Path(args.yoloworld_dir),
                       scenario=args.data_split_scenario,
                       apply_nms=args.apply_nms)