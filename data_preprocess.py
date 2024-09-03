
import pickle
import os
import json
import h5py
from tqdm import tqdm
import argparse
import numpy as np
from pathlib import Path
import torch
import torch.nn.functional as F

from utils.data_util import NuScenesClasses, NuScenesClassesBase
from utils.util import load_clip, extract_clip_feature, farthest_point_sampling, zero_pad_point_cloud

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

def save_dict_to_hdf5(group, data_dict):
    for key, value in data_dict.items():
        if isinstance(value, dict):
            # Save each dictionary as a group
            subgroup = group.create_group(key)
            save_dict_to_hdf5(subgroup, value)
        elif isinstance(value, list):
            # If the value is a list, create a group for the list and iterate over the elements
            list_group = group.create_group(key)
            for i, item in enumerate(value):
                if isinstance(item, dict):
                    # Save each dictionary in the list as a subgroup
                    item_group = list_group.create_group(str(i))
                    save_dict_to_hdf5(item_group, item)
                else:
                    # Save the item directly if it's not a dict
                    list_group.create_dataset(str(i), data=np.array(item))
        else:
            # Save non-dict and non-list items directly
            group.create_dataset(key, data=np.array(value))

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
    frame_det_data['points'] = [frame_det_data['points'][i] for i in index]
    frame_det_data['pt_features'] = [frame_det_data['pt_features'][i] for i in index]

    return frame_det_data

def drop_empty_detections(data):
    indices = []
    for i, box_points in enumerate(data['dets']['points']):
        if box_points.size != 0:
            indices.append(i)
    data['dets']['translation'] = data['dets']['translation'][indices]
    data['dets']['size'] = data['dets']['size'][indices]
    data['dets']['yaw'] = data['dets']['yaw'][indices]
    data['dets']['velocity'] = data['dets']['velocity'][indices]
    data['dets']['class'] = data['dets']['class'][indices]
    data['dets']['score'] = data['dets']['score'][indices]
    data['dets']['points'] =[data['dets']['points'][i] for i in indices]
    data['dets']['pt_features'] = [data['dets']['pt_features'][i] for i in indices]

    return data

def drop_features(data):

        del data['dets']['embedding']  

        if 'sam_points' in data['dets']:
            del data['dets']['sam_points']  
            del data['dets']['sam_pt_features']
            del data['dets']['nbr_points'] 

        return data

def obtain_openscene_labels(split, clip, data):
    if split == "training":
        labelset = list(NuScenesClassesBase.keys())
        labelset_prompt = [ "a " + label + " in a scene" for label in labelset]
    else:
        labelset = list(NuScenesClasses.keys())
        labelset_prompt = [ "a " + label + " in a scene" for label in labelset]

    text_features = extract_clip_feature(labelset_prompt, clip)
    print(text_features.size())
    class_openscene = []
    embedding = []
    # class_pts = []
    # class_pt_feats = []
    for i, box_features in enumerate(data['dets']['pt_features']):

        pred = torch.from_numpy(box_features).to('cuda:0').half() @ text_features.t()
        pred = F.softmax(pred, dim=-1)
        logits_pred = torch.max(pred, 1)[1].cpu().numpy()

        # top_vals, val_ind = torch.topk(pred, 2)
        # top_vals = top_vals.cpu().detach().numpy()
        # val_ind = val_ind.cpu().detach().numpy()
        # ratios = top_vals[:, 1] / top_vals[:, 0]
        # logits_pred2 = np.where(ratios > 0.8, val_ind[:, 1], val_ind[:, 0])

        unique_values, counts = np.unique(logits_pred, return_counts=True)

        # points = []
        # points_ft = []
        # for ind in range(len(labelset)):
        #     if ind in unique_values:
        #         pts = data['dets']['points'][i][logits_pred == ind]
        #         pt_feats = data['dets']['pt_features'][i][logits_pred == ind]
        #         points.append(np.mean(pts, axis=0))
        #         points_ft.append(np.mean(pt_feats, axis=0))
        #     else:
        #         points.append(np.array([0.0, 0.0, 0.0]))
        #         points_ft.append(np.zeros(768))
        # class_pts.append(points)
        # class_pt_feats.append(points_ft)

        index_label = np.argmax(counts)
        label = labelset[unique_values[index_label]]
        # class_openscene.append(int(NuScenesClasses[label]))

        # unique_values, counts = np.unique(logits_pred2, return_counts=True)
        # index_label = np.argmax(counts)
        # label = labelset[unique_values[index_label]]
        # print("Lowe's maj: ",label)
        # print(list(NuScenesClasses.keys())[int(data['dets']['class'][i])])

        label_mask = logits_pred == unique_values[index_label]
        masked_pts = data['dets']['pt_features'][i][label_mask]
        max_score = np.argmax(pred.cpu().detach().numpy()[label_mask, unique_values[index_label]])
        embedding.append(masked_pts[max_score])

    # data['dets']['class_openscene'] = np.array(class_openscene, dtype=np.int32)
    data['dets']['embedding'] = np.array(embedding, dtype=np.float32)
    # data['dets']['points'] = np.array(class_pts, dtype=np.float32)
    # data['dets']['pt_features'] = np.array(class_pt_feats, dtype=np.float32)

    return data

def sampling_detection_points(data):
    point_samples = 32
    # nbr_points = []
    embedding = []
    # sampled_points = []
    # sampled_pt_features = [ ]
    for i, box_points in enumerate(data['dets']['points']):
        if box_points.shape[0] > point_samples:
            embedding.append((np.sum(data['dets']['pt_features'][i], axis=0))/box_points.shape[0])
            # indices = farthest_point_sampling(box_points, point_samples)
            # sampled_points.append(box_points[indices])
            # sampled_pt_features.append(data['dets']['pt_features'][i][indices])
            # nbr_points.append(point_samples)
            
        else:
            # nbr_points.append(box_points.shape[0])
            embedding.append((np.sum(data['dets']['pt_features'][i], axis=0))/box_points.shape[0])
            # sampled_points.append(zero_pad_point_cloud(box_points, point_samples))
            # sampled_pt_features.append(zero_pad_point_cloud(data['dets']['pt_features'][i],
                                                                    # point_samples))
    
    # data['dets']['nbr_points'] = np.array(nbr_points, dtype=np.float32)
    data['dets']['embedding_avg'] = np.array(embedding, dtype=np.float32)
    # data['dets']['sam_points'] = np.stack(sampled_points).astype(np.float32)
    # data['dets']['sam_pt_features'] = np.stack(sampled_pt_features).astype(np.float32)

    return data

def base_ground_truth(split, data):

    if split == 'training':
        indices = []
        all_labels = list(NuScenesClasses.keys())
        for i, class_label in enumerate(data['gts']['class']):
            if all_labels[class_label] in NuScenesClassesBase.keys():
                indices.append(i)
        
        for item in data['gts']:
            data['gts'][item] = data['gts'][item][indices]
        
        indices = []
        for i, class_label in enumerate(data['dets']['class']):
            if all_labels[class_label] in NuScenesClassesBase.keys():
                indices.append(i)
        
        for item in data['dets']:
            data['dets'][item] = data['dets'][item][indices]

    return data

def drop_background(data):

    indices = []
    for i, class_label in enumerate(data['dets']['yolo_class']):
        if class_label != 7:
            indices.append(i)
    
    for item in data['dets']:
        data['dets'][item] = data['dets'][item][indices]

    return data

def convert_emb_class(data, clip):
    labelset = list(NuScenesClasses.keys()) + ['background']
    text_features = extract_clip_feature(labelset, clip).detach().cpu().numpy()
    yolo_class = []
    for i, emb in enumerate(data['dets']['embedding']):
        index = np.where(np.all(text_features == emb, axis=1))[0]
        # index = next((i for i, clip_emb in enumerate(text_features) if np.array_equal(clip_emb, emb)), None)
        print(data['dets']['class'][i], index)
        yolo_class.append(index)
    data['dets']['yolo_class'] = np.array(yolo_class, dtype=np.int)
    return data


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

def main(output_dir, save_dir):
    # clip = load_clip()

    for split in [ "validation"]:
        split_path = output_dir / split
        save_output = save_dir
        save_output.mkdir(parents=True, exist_ok=True)
        save_output = save_dir / (split + '.h5')
        split_data = {}
        for scene_id, scene in enumerate(sorted(os.listdir(split_path))):
            print("scene ID", scene_id)
            scene_path = split_path / scene
            scene_result = []
            frames = sorted(os.listdir(scene_path))
            for frame_id, frame in enumerate(tqdm(frames)):
                frame_pkl = scene_path / frame
                f = open(frame_pkl, 'rb') 
                frame = pickle.load(f)
                
                # if split == 'training':
                #     frame_data['dets'] = simpletrack_nms(frame_data['dets'], iou_threshold=0.1)
                # frame_data = drop_empty_detections(frame_data)
                # frame_data = obtain_openscene_labels(split, clip, frame_data)
                # frame_data = sampling_detection_points(frame_data)
                # frame_data = drop_features(frame_data)
                # frame_data = base_ground_truth(split, frame_data)
                # frame_data = convert_emb_class(frame_data, clip)
                # frame_data = drop_features(frame_data)
                content = {'dets': frame['dets'],
                   'gts': frame['gts'],
                   'num_dets': len(frame['dets']['translation']),
                   'num_gts': len(frame['gts']['translation']),
                   'ego_translation': frame['ego_translation'],
                   'timestamp': frame['timestamp'],
                   'token': frame['token']
                  }
                scene_result.append(content)
                print(np.min(content['dets']['yolo_score']))
                # for i, each in enumerate(content['dets']['score']):
                #     print(each, content['dets']['yolo_score'][i])
            # split_data[f'{scene_id:04d}'] = scene_result
        
                        

                        

if __name__ == '__main__':
    
    args = argparse.ArgumentParser(description='Nuscenes data preprocessing')
    args.add_argument('--output_dir', default=None, type=str,
                      help='Directory where openscene inference pickle files are stored')
    args.add_argument('--save_dir', default=None, type=str,
                      help='Directory where processed pickle files are stored')
    args = args.parse_args()

    main(output_dir=Path(args.output_dir), save_dir=Path(args.save_dir))

