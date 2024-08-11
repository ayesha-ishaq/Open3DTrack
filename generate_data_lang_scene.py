import pickle
import os
import cv2
from copy import deepcopy
from tqdm import tqdm
import argparse
import numpy as np
from PIL import Image
from pathlib import Path
from collections import OrderedDict
from pyquaternion import Quaternion

from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes
from nuscenes.utils.data_classes import LidarPointCloud, Box
from shapely.geometry import MultiPoint, box, LineString
from nuscenes.utils.geometry_utils import view_points, points_in_box
from nuscenes.eval.common.loaders import load_prediction
from nuscenes.eval.detection.data_classes import DetectionBox
from nuscenes.eval.common.utils import quaternion_yaw
from nuscenes.eval.tracking.utils import category_to_tracking_name

from utils.data_util import NuScenesClasses, NuScenesClassesBase, category_to_tracking_name_base
import torch
import clip

from utils.util import load_clip, extract_clip_feature

from tensorflow import io
import tensorflow as tf2
import tensorflow.compat.v1 as tf

from utils.data_util import NuScenesClasses

def read_bytes(path):
    '''Read bytes for OpenSeg model running.'''

    with io.gfile.GFile(path, 'rb') as f:
        file_bytes = f.read()
    return file_bytes

def load_openseg(saved_model_dir):

    openseg_model = tf2.saved_model.load(saved_model_dir, tags=[tf.saved_model.tag_constants.SERVING],)

    return openseg_model

def extract_openseg_img_feature(img_dir, openseg_model, text_emb, img_size=None, regional_pool=True):
    '''Extract per-pixel OpenSeg features.'''

    # load RGB image
    np_image_string = read_bytes(img_dir)
    # run OpenSeg
    results = openseg_model.signatures['serving_default'](
            inp_image_bytes=tf.convert_to_tensor(np_image_string),
            inp_text_emb=text_emb)
    # ppixel_ave_feat

    img_info = results['image_info']
    crop_sz = [
        int(img_info[0, 0] * img_info[2, 0]),
        int(img_info[0, 1] * img_info[2, 1])]
    if regional_pool:
        image_embedding_feat = results['ppixel_ave_feat'][:, :crop_sz[0], :crop_sz[1]]
    else:
        image_embedding_feat = results['image_embedding_feat'][:, :crop_sz[0], :crop_sz[1]]
    if img_size is not None:
        feat_2d = tf.cast(tf.image.resize_nearest_neighbor(
            image_embedding_feat, img_size, align_corners=True)[0], dtype=tf.float32).numpy()
    else:
        feat_2d = tf.cast(image_embedding_feat[[0]], dtype=tf.float32).numpy()

    return feat_2d

def post_process_coords(corner_coords, imsize=(1600, 900)):
    """
    Get the intersection of the convex hull of the reprojected bbox corners and the image canvas, return None if no
    intersection.
    :param corner_coords: Corner coordinates of reprojected bounding box.
    :param imsize: Size of the image canvas.
    :return: Intersection of the convex hull of the 2D box corners and the image canvas.
    """
    polygon_from_2d_box = MultiPoint(corner_coords).convex_hull
    img_canvas = box(0, 0, imsize[0], imsize[1])

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

def map_pointcloud_to_image(nusc, current_sample, openseg_model,
                            min_dist=0.0):
    """
    Given a point sensor (lidar/radar) token and camera sample_data token, load pointcloud and map it to the image
    plane.
    :param pointsensor_token: Lidar/radar sample_data token.
    :param camera_token: Camera sample_data token.
    :param min_dist: Distance from the camera below which points are discarded.
    :return (pointcloud <np.float: 2, n)>, coloring <np.float: n>, image <Image>).
    """

    cameras = ["CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT", "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"]

    text_emb = tf.zeros([1, 1, 768])

    pointsensor = nusc.get('sample_data', current_sample['data']['LIDAR_TOP'])
    pcl_path = os.path.join(nusc.dataroot, pointsensor['filename'])
    pcl = LidarPointCloud.from_file(pcl_path)

    # world_point = deepcopy(pc.points)
    # Points live in the point sensor frame. So they need to be transformed via global to the image plane.
    # First step: transform the pointcloud to the ego vehicle frame for the timestamp of the sweep.
    cs_record = nusc.get('calibrated_sensor', pointsensor['calibrated_sensor_token'])
    pcl.rotate(Quaternion(cs_record['rotation']).rotation_matrix)
    pcl.translate(np.array(cs_record['translation']))

    # Second step: transform from ego to the global frame.
    poserecord = nusc.get('ego_pose', pointsensor['ego_pose_token'])
    pcl.rotate(Quaternion(poserecord['rotation']).rotation_matrix)
    pcl.translate(np.array(poserecord['translation']))

    global_points = deepcopy(pcl.points)
    global_embeddings = np.zeros((768, global_points.shape[-1]))

    for cam in cameras:
        pc = deepcopy(pcl)
        camera = nusc.get('sample_data', current_sample['data'][cam])

        # Third step: transform from global into the ego vehicle frame for the timestamp of the image.
        poserecord = nusc.get('ego_pose', camera['ego_pose_token'])
        pc.translate(-np.array(poserecord['translation']))
        pc.rotate(Quaternion(poserecord['rotation']).rotation_matrix.T)

        # Fourth step: transform from ego into the camera.
        cs_record = nusc.get('calibrated_sensor', camera['calibrated_sensor_token'])
        pc.translate(-np.array(cs_record['translation']))
        pc.rotate(Quaternion(cs_record['rotation']).rotation_matrix.T)

        # Fifth step: actually take a "picture" of the point cloud.
        # Grab the depths (camera frame z axis points away from the camera).
        depths = pc.points[2, :]

        scale = 640/1600 #openseg img size, original img size

        intrinsic = np.array(cs_record['camera_intrinsic'])
        intrinsic[:2, :] = intrinsic[:2, :] * scale
        # Take the actual picture (matrix multiplication with camera-matrix + renormalization).
        points = view_points(pc.points[:3, :], intrinsic, normalize=True)

        # Remove points that are either outside or behind the camera. Leave a margin of 1 pixel for aesthetic reasons.
        # Also make sure points are at least 1m in front of the camera to avoid seeing the lidar points on the camera
        # casing for non-keyframes which are slightly out of sync.
        mask = np.ones(depths.shape[0], dtype=bool)
        mask = np.logical_and(mask, depths > min_dist)
        mask = np.logical_and(mask, points[0, :] > 1)
        mask = np.logical_and(mask, points[0, :] < 1600*scale - 1)
        mask = np.logical_and(mask, points[1, :] > 1)
        mask = np.logical_and(mask, points[1, :] < 900*scale - 1)
        points = points[:, mask].astype(np.int)

        img_path = os.path.join(nusc.dataroot, camera['filename'])
        img_embeddings = extract_openseg_img_feature(img_path, openseg_model, text_emb)

        global_embeddings[:, mask] = img_embeddings[points[1, :], points[0, :], :].T

    non_zero_mask = ~np.all(global_embeddings == 0, axis=0) 
    print(global_embeddings.shape, np.sum(non_zero_mask)) 
    global_embeddings = global_embeddings[:, non_zero_mask]
    global_points = global_points[:, non_zero_mask] 
        
    return global_points, global_embeddings

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

def generate_nusc_seq_data(nusc, det_boxes, scenes, sequences_by_name, output, split, openseg_path, apply_nms=False):

    print('Generating detection and ground truth sequences...')
    result = []
    clip = load_clip()
    labelset = list(NuScenesClasses.keys())
    text_features = extract_clip_feature(labelset, clip)
    openseg_model = load_openseg(openseg_path)
    
    for scene_id, scene_name in enumerate(tqdm(scenes)):
        if scene_id >= 88:
            scene = sequences_by_name[scene_name]
            first_token = scene['first_sample_token']
            last_token = scene['last_sample_token']
            scene_desc = scene['description']
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
    
                if apply_nms:
                    frame_dets_dict = simpletrack_nms(frame_dets_dict, iou_threshold=0.1)
                    
                # get 2D bounding box for nms detections only
                frame_dets = np.zeros(len(frame_dets_dict['translation']), dtype=bool)
                avg_embedding = np.zeros((len(frame_dets_dict['translation']), 768))
                frame_det_pts_features = []

                pc_transform = nusc.get('calibrated_sensor', lidar_top_data['calibrated_sensor_token'])
                global_points, mapped_emb = map_pointcloud_to_image(nusc, current_sample,
                                                                    openseg_model)

                for nms_dets_index in range(len(frame_dets_dict['translation'])):

                    box = Box(frame_dets_dict['translation'][nms_dets_index],
                            frame_dets_dict['size'][nms_dets_index], 
                            Quaternion(frame_dets_dict['rotation'][nms_dets_index]))
                    # # Move them to the ego-pose frame.
                    # box.translate(-np.array(ego_pose['translation']))
                    # box.rotate(Quaternion(ego_pose['rotation']).inverse)

                    # # Move them to the calibrated sensor frame.
                    # box.translate(-np.array(pc_transform['translation']))
                    # box.rotate(Quaternion(pc_transform['rotation']).inverse)

                    points_mask = points_in_box(box, global_points[:3, :])
                    embeddings = mapped_emb[:, points_mask]

                    if embeddings.size != 0:   
                        embs = torch.from_numpy(embeddings.T)
                        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                        # Assuming embs and text_features are your tensors
                        embs = embs.to(device)
                        pred = embs.half() @ text_features.t()
                        logits_pred = torch.max(pred, 1)[1].cpu()

                        print('orig class', frame_dets_dict['class'][nms_dets_index])
                        print('emb classes', logits_pred)

                        pred = torch.argmax(torch.mean(embs, dim=0).half() @ text_features.t()).cpu()

                        print('avg class', pred)              
                        frame_dets[nms_dets_index] = 1
                        frame_det_pts_features.append(embeddings)
                        avg_embedding[nms_dets_index] = np.mean(embeddings, axis=-1)
                
                print(len(frame_dets_dict['translation']))
                for key in frame_dets_dict:
                    frame_dets_dict[key] = frame_dets_dict[key][frame_dets]
                
                print(len(frame_dets_dict['translation']))
                frame_dets_dict['embedding'] = avg_embedding.astype(np.float32) # [N, 768]
                frame_dets_dict['points'] = frame_det_pts_features

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
    
            # write_data_per_scene(scene_result, output)
            result.append(scene_result)
        
    print('Done generating.')
    print('======')
    
    return result

def generate_nusc_data(version, dataset_dir, detection_dir, output_dir, openseg_path, apply_nms=False):

    dataset_dir = dataset_dir 
    train_result_file = detection_dir / "train.json"
    val_result_file = detection_dir / "val.json"
    test_result_file = detection_dir / "test.json"

    version_fullname = version
    if version == "v1.0":
        version_fullname += '-trainval'
    nusc = NuScenes(version=version_fullname, dataroot=dataset_dir, verbose=True)
    # nusc_test = NuScenes(version='v1.0-test', dataroot=dataset_dir, verbose=True)
    sequences_by_name = {scene["name"]: scene for scene in nusc.scene}
    # sequences_by_name.update({scene["name"]: scene for scene in nusc_test.scene})
    splits_to_scene_names = create_splits_scenes()

    train_split = 'train' if version == "v1.0" else 'mini_train'
    val_split = 'val' if version == "v1.0" else 'mini_val'
    test_split = 'test'
    train_scenes = splits_to_scene_names[train_split]
    val_scenes = splits_to_scene_names[val_split]
    test_scenes = splits_to_scene_names[test_split]

    result_files = [train_result_file, val_result_file]
    scenes = [train_scenes, val_scenes]
    output_dirs = [output_dir / 'training', output_dir / 'validation']
    splits = ['training', 'validation']


    # Train and validation split
    for result_file, scene, output, split in zip(result_files, scenes, output_dirs, splits):
        # output.mkdir(parents=True, exist_ok=True)
        if split == 'validation':
          print('Loading Nusences 3d detctions...')
          det_boxes, _ = load_prediction(result_file, 10000, DetectionBox, verbose=True)
          print('======')
  
          data = generate_nusc_seq_data(nusc, det_boxes, scene, sequences_by_name, output, split, openseg_path, apply_nms)

        # write_data(data, output)
    
    # Test split
    # print('Loading Nusences 3d detctions...')
    # det_boxes, _ = load_prediction(test_result_file, 10000, DetectionBox, verbose=True)
    # print('======')

    # data = generate_nusc_seq_data(nusc_test, det_boxes, test_scenes, sequences_by_name, apply_nms)

    # write_data(data, output_dir / 'testing')


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='Nuscenes data preprocessing')
    args.add_argument('--dataset_dir', default=None, type=str,
                      help='Directory where nuScenes dataset is stored')
    args.add_argument('--version', default="v1.0", type=str,
                      help='Version of nuScenes dataset')
    args.add_argument('--detection_dir', default=None, type=str,
                      help='Directory where detection results are stored')
    args.add_argument('--output_dir', default=None, type=str,
                      help='Directory where preprocessed pickle files will be stored')
    args.add_argument('--openseg_dir', default=None, type=str,
                      help='Directory where openseg will be stored')
    args.add_argument('--apply_nms', action='store_true',
                      help='Whether to apply a Non-Maximum Suppression')
    args = args.parse_args()

    generate_nusc_data(version=args.version,
                       dataset_dir=Path(args.dataset_dir),
                       detection_dir=Path(args.detection_dir),
                       output_dir=Path(args.output_dir),
                       openseg_path=args.openseg_dir,
                       apply_nms=args.apply_nms)