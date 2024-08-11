
import pickle
import os
from tqdm import tqdm
import argparse
import numpy as np
from pathlib import Path

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

def main(output_dir):
    for split in ["training", "validation"]:
        split_path = output_dir / split
        for scene_id, scene in enumerate(sorted(os.listdir(split_path))):
            print("scene ID", scene_id)
            scene_path = split_path / scene
            scene_result = []
            frames = sorted(os.listdir(scene_path))
            for frame_id, frame in enumerate(tqdm(frames)):
                frame_pkl = scene_path / frame
                f = open(frame_pkl, 'rb') 
                frame_data = pickle.load(f)
                frame_dets_dict = frame_data["dets"]
                frame_dets_dict = simpletrack_nms(frame_dets_dict, iou_threshold=0.1)
                frame_result = {'detections': frame_dets_dict,
                            'ground_truths': frame_data['gts'],
                            'num_dets': len(frame_dets_dict['translation']), # int: N
                            'num_gts': frame_data['num_gts'], # int: M
                            'scene_id': scene_id,
                            'frame_id': frame_id,
                            'ego_translation': frame_data['ego_translation'],
                            'timestamp': frame_data['timestamp'],
                            'sample_token': frame_data['token']
                            }
                scene_result.append(frame_result)
            write_data_per_scene(scene_result, split_path)




if __name__ == '__main__':
    
    args = argparse.ArgumentParser(description='Nuscenes data preprocessing')
    args.add_argument('--output_dir', default=None, type=str,
                      help='Directory where preprocessed pickle files are stored')
    args = args.parse_args()

    main(output_dir=Path(args.output_dir))