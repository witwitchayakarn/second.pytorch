import json
import pickle
import time
import random
from copy import deepcopy
from functools import partial
from pathlib import Path
import subprocess
from multiprocessing import Pool

import fire
import numpy as np

from pyquaternion import Quaternion

from second.core import box_np_ops
from second.core import preprocess as prep
from second.data import kitti_common as kitti
from second.data.dataset import Dataset, register_dataset
from second.utils.eval import get_coco_eval_result, get_official_eval_result
from second.utils.progress_bar import progress_bar_iter as prog_bar
from second.utils.timer import simple_timer


@register_dataset
class LyftDataset(Dataset):
    NumPointFeatures = 4  # xyz, timestamp. set 4 to use kitti pretrain

    def __init__(self,
                 root_path,
                 info_path,
                 class_names=None,
                 prep_func=None,
                 num_point_features=None):

        self._root_path = Path(root_path)
        self._class_names = class_names
        self._prep_func = prep_func

        with open(info_path, 'rb') as f:
            data = pickle.load(f)

        self._nusc_infos = data['infos']
        self._metadata = data['metadata']

        self._nusc_infos = list(sorted(self._nusc_infos, key=lambda e: e['timestamp']))

        self._with_velocity = False
        self._cache_of_ground_truth_annotations = None

    def __len__(self):
        return len(self._nusc_infos)

    def __getitem__(self, idx):
        input_dict = self.get_sensor_data(idx)
        example = self._prep_func(input_dict=input_dict)
        example['metadata'] = input_dict['metadata']
        if 'anchors_mask' in example:
            example['anchors_mask'] = example['anchors_mask'].astype(np.uint8)
        return example

    def get_sensor_data(self, query):
        idx = query
        read_test_image = False
        if isinstance(query, dict):
            assert 'lidar' in query
            idx = query['lidar']['idx']
            read_test_image = 'cam' in query

        info = self._nusc_infos[idx]
        res = {
            'lidar': {
                'type': 'lidar',
                'points': None,
            },
            'metadata': {
                'token': info['token']
            },
        }

        lidar_path = Path(info['lidar_path'])
        points = np.fromfile(str(lidar_path), dtype=np.float32, count=-1)
        if points.shape[0] % 5 != 0:
            points = points[:int(points.shape[0]/5)*5]
        points = points.reshape([-1, 5])
        points[:, 3] /= 255
        points[:, 4] = 0
        sweep_points_list = [points]
        ts = info['timestamp'] / 1e6
        for sweep in info['sweeps']:
            points_sweep = np.fromfile(str(sweep['lidar_path']),
                                       dtype=np.float32,
                                       count=-1)
            if points_sweep.shape[0] % 5 != 0:
                points_sweep = points_sweep[:int(points_sweep.shape[0]/5)*5]
            points_sweep = points_sweep.reshape([-1, 5])
            sweep_ts = sweep['timestamp'] / 1e6
            points_sweep[:, 3] /= 255
            points_sweep[:, :3] = points_sweep[:, :3] @ sweep['sweep2lidar_rotation'].T
            points_sweep[:, :3] += sweep['sweep2lidar_translation']
            points_sweep[:, 4] = ts - sweep_ts
            sweep_points_list.append(points_sweep)

        points = np.concatenate(sweep_points_list, axis=0)[:, [0, 1, 2, 4]]
        res['lidar']['points'] = points

        if read_test_image:
            if Path(info['cam_front_path']).exists():
                with open(str(info['cam_front_path']), 'rb') as f:
                    image_str = f.read()
            else:
                image_str = None
            res['cam'] = {
                'type': 'camera',
                'data': image_str,
                'datatype': Path(info['cam_front_path']).suffix[1:],
            }

        if 'gt_boxes' in info:
            gt_boxes = info['gt_boxes']
            if self._with_velocity:
                gt_velocity = info['gt_velocity']
                nan_mask = np.isnan(gt_velocity[:, 0])
                gt_velocity[nan_mask] = [0.0, 0.0]
                gt_boxes = np.concatenate([gt_boxes, gt_velocity], axis=-1)
            res['lidar']['annotations'] = {
                'boxes': gt_boxes,
                'names': info['gt_names'],
            }

        return res

    @property
    def ground_truth_annotations(self):

        if 'gt_boxes' not in self._nusc_infos[0]:
            return None

        if self._cache_of_ground_truth_annotations is not None:
            return self._cache_of_ground_truth_annotations

        gt_annos = []
        for info in self._nusc_infos:
            boxes = _info_to_nusc_box(info)
            boxes = _lidar_nusc_box_to_global(info, boxes)

            for i, box in enumerate(boxes):
                gt_annos.append({
                    'sample_token': info['token'],
                    'translation': box.center.tolist(),
                    'size': box.wlh.tolist(),
                    'rotation': box.orientation.elements.tolist(),
                    'name': box.name
                })

        self._cache_of_ground_truth_annotations = gt_annos
        return self._cache_of_ground_truth_annotations

    def evaluation(self, detections, output_dir):
        gt_annos = self.ground_truth_annotations
        if gt_annos is None:
            return None

        token2info = {}
        for info in self._nusc_infos:
            token2info[info["token"]] = info

        predictions = []
        for det in detections:
            boxes = _second_det_to_nusc_box(det)
            boxes = _lidar_nusc_box_to_global(token2info[det['metadata']['token']],
                                              boxes)
            for i, box in enumerate(boxes):
                predictions.append({
                    'sample_token': det['metadata']['token'],
                    'translation': box.center.tolist(),
                    'size': box.wlh.tolist(),
                    'rotation': box.orientation.elements.tolist(),
                    'name': self._class_names[box.label],
                    'score': box.score
                })

        iou_threshold = 0.5
        average_precisions = _get_average_precisions(gt_annos,
                                                     predictions,
                                                     self._class_names,
                                                     iou_threshold)

        result = f"Lyft Evaluation\n"
        detail = {}

        mAP = np.mean(average_precisions)

        print('Average per class mean average precision = ', mAP)
        result += f"Average per class mean average precision = {mAP}\n"
        detail['mAP'] = mAP

        for name, ap in zip(self._class_names, average_precisions.tolist()):
            print(f"{name}: {ap}")
            result += f"{name}: {ap}\n"
            detail[name] = ap

        return {
            "results": {
                "lyft": result
            },
            "detail": {
                "lyft": detail
            },
        }


def _second_det_to_nusc_box(detection):
    from lyft_dataset_sdk.utils.data_classes import Box

    box3d = detection["box3d_lidar"].detach().cpu().numpy()
    scores = detection["scores"].detach().cpu().numpy()
    labels = detection["label_preds"].detach().cpu().numpy()

    box3d[:, 6] = -box3d[:, 6] - np.pi / 2

    box_list = []
    for i in range(box3d.shape[0]):
        quat = Quaternion(axis=[0, 0, 1], radians=box3d[i, 6])
        velocity = (np.nan, np.nan, np.nan)
        if box3d.shape[1] == 9:
            velocity = (*box3d[i, 7:9], 0.0)
            # velo_val = np.linalg.norm(box3d[i, 7:9])
            # velo_ori = box3d[i, 6]
            # velocity = (velo_val * np.cos(velo_ori), velo_val * np.sin(velo_ori), 0.0)
        box = Box(box3d[i, :3],
                  box3d[i, 3:6],
                  quat,
                  label=labels[i],
                  score=scores[i],
                  velocity=velocity)
        box_list.append(box)
    return box_list


def _info_to_nusc_box(info):
    from lyft_dataset_sdk.utils.data_classes import Box

    box3d = info['gt_boxes'].copy()
    names = info['gt_names'].copy()

    box3d[:, 6] = -box3d[:, 6] - np.pi / 2

    box_list = []
    for i in range(box3d.shape[0]):
        quat = Quaternion(axis=[0, 0, 1], radians=box3d[i, 6])
        velocity = (np.nan, np.nan, np.nan)
        box = Box(box3d[i, :3],
                  box3d[i, 3:6],
                  quat,
                  name=names[i],
                  velocity=velocity)
        box_list.append(box)
    return box_list


def _lidar_nusc_box_to_global(info, boxes):
    box_list = []
    for box in boxes:
        # Move box to ego vehicle coord system
        box.rotate(Quaternion(info['lidar2ego_rotation']))
        box.translate(np.array(info['lidar2ego_translation']))

        # Move box to global coord system
        box.rotate(Quaternion(info['ego2global_rotation']))
        box.translate(np.array(info['ego2global_translation']))
        box_list.append(box)

    return box_list


def _get_average_precisions(gt: list,
                            predictions: list,
                            class_names: list,
                            iou_threshold: float) -> np.array:
    assert 0 <= iou_threshold <= 1

    from lyft_dataset_sdk.eval.detection.mAP_evaluation import group_by_key
    gt_by_class_name = group_by_key(gt, "name")
    pred_by_class_name = group_by_key(predictions, "name")

    pool = Pool(8)
    pool_inputs = [(gt_by_class_name[name],
                    pred_by_class_name[name],
                    iou_threshold,
                    name) for name in class_names if name in pred_by_class_name]
    pool_results = pool.starmap(_recall_precision, pool_inputs)

    average_precisions = np.zeros(len(class_names))
    for class_name, average_precision in pool_results:
        average_precisions[class_names.index(class_name)] = average_precision

    return average_precisions


def _recall_precision(gt, pred, th, class_name):
    from lyft_dataset_sdk.eval.detection.mAP_evaluation import recall_precision
    recalls, precisions, average_precision = recall_precision(gt, pred, th)
    return class_name, average_precision


def _fill_trainval_infos(lyft_ds,
                         train_scenes,
                         val_scenes,
                         do_test=False,
                         max_sweeps=10):

    train_nusc_infos = []
    val_nusc_infos = []
    for sample in prog_bar(lyft_ds.sample):
        lidar_token = sample['data']['LIDAR_TOP']
        cam_front_token = sample['data']['CAM_FRONT']

        sd_rec = lyft_ds.get('sample_data', lidar_token)
        cs_record = lyft_ds.get('calibrated_sensor', sd_rec['calibrated_sensor_token'])
        pose_record = lyft_ds.get('ego_pose', sd_rec['ego_pose_token'])

        lidar_path, boxes, _ = lyft_ds.get_sample_data(lidar_token)
        cam_path, _, cam_intrinsic = lyft_ds.get_sample_data(cam_front_token)

        assert Path(lidar_path).exists(), (
            'you must download all trainval data, key-frame only dataset performs far worse than sweeps.'
        )

        info = {
            'lidar_path': lidar_path,
            'cam_front_path': cam_path,
            'token': sample['token'],
            'sweeps': [],
            'lidar2ego_translation': cs_record['translation'],
            'lidar2ego_rotation': cs_record['rotation'],
            'ego2global_translation': pose_record['translation'],
            'ego2global_rotation': pose_record['rotation'],
            'timestamp': sample['timestamp'],
        }

        l2e_r = info['lidar2ego_rotation']
        l2e_t = info['lidar2ego_translation']
        e2g_r = info['ego2global_rotation']
        e2g_t = info['ego2global_translation']
        l2e_r_mat = Quaternion(l2e_r).rotation_matrix
        e2g_r_mat = Quaternion(e2g_r).rotation_matrix

        sweeps = []
        while len(sweeps) < max_sweeps:
            if not sd_rec['prev'] == '':
                sd_rec = lyft_ds.get('sample_data', sd_rec['prev'])
                cs_record = lyft_ds.get('calibrated_sensor',
                                        sd_rec['calibrated_sensor_token'])
                pose_record = lyft_ds.get('ego_pose', sd_rec['ego_pose_token'])
                lidar_path = lyft_ds.get_sample_data_path(sd_rec['token'])

                sweep = {
                    'lidar_path': lidar_path,
                    'sample_data_token': sd_rec['token'],
                    'lidar2ego_translation': cs_record['translation'],
                    'lidar2ego_rotation': cs_record['rotation'],
                    'ego2global_translation': pose_record['translation'],
                    'ego2global_rotation': pose_record['rotation'],
                    'timestamp': sd_rec['timestamp']
                }

                l2e_r_s = sweep['lidar2ego_rotation']
                l2e_t_s = sweep['lidar2ego_translation']
                e2g_r_s = sweep['ego2global_rotation']
                e2g_t_s = sweep['ego2global_translation']
                # sweep->ego->global->ego'->lidar
                l2e_r_s_mat = Quaternion(l2e_r_s).rotation_matrix
                e2g_r_s_mat = Quaternion(e2g_r_s).rotation_matrix

                R = (l2e_r_s_mat.T @ e2g_r_s_mat.T) @ (
                    np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
                T = (l2e_t_s @ e2g_r_s_mat.T + e2g_t_s) @ (
                    np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
                T -= e2g_t @ (np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(
                    l2e_r_mat).T) + l2e_t @ np.linalg.inv(l2e_r_mat).T
                sweep['sweep2lidar_rotation'] = R.T  # points @ R.T + T
                sweep['sweep2lidar_translation'] = T
                sweeps.append(sweep)
            else:
                break
        info['sweeps'] = sweeps

        if not do_test:
            annotations = [lyft_ds.get('sample_annotation', token) for token in sample['anns']]
            locs = np.array([b.center for b in boxes]).reshape(-1, 3)
            dims = np.array([b.wlh for b in boxes]).reshape(-1, 3)
            rots = np.array([b.orientation.yaw_pitch_roll[0] for b in boxes]).reshape(-1, 1)
            velocity = np.array([lyft_ds.box_velocity(token)[:2] for token in sample['anns']])

            # convert velo from global to lidar
            for i in range(len(boxes)):
                velo = np.array([*velocity[i], 0.0])
                velo = velo @ np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T
                velocity[i] = velo[:2]

            names = np.array([b.name for b in boxes])

            # we need to convert rot to SECOND format.
            # change the rot format will break all checkpoint, so...
            gt_boxes = np.concatenate([locs, dims, -rots - np.pi / 2], axis=1)
            assert len(gt_boxes) == len(annotations), f'{len(gt_boxes)}, {len(annotations)}'

            info['gt_boxes'] = gt_boxes
            info['gt_names'] = names
            info['gt_velocity'] = velocity.reshape(-1, 2)
            info['num_lidar_pts'] = np.array([a['num_lidar_pts'] for a in annotations])
            info['num_radar_pts'] = np.array([a['num_radar_pts'] for a in annotations])

        if sample['scene_token'] in train_scenes:
            train_nusc_infos.append(info)
        else:
            val_nusc_infos.append(info)

    return train_nusc_infos, val_nusc_infos


def create_nuscenes_infos(root_path, version='trainval', max_sweeps=10):
    available_vers = ['trainval', 'test']
    assert version in available_vers

    do_test = 'test' in version

    from lyft_dataset_sdk.lyftdataset import LyftDataset as LyftDatasetSDKLyftDataset
    lyft_ds = LyftDatasetSDKLyftDataset(data_path=root_path,
                                        json_path=Path(root_path) / 'data',
                                        verbose=True)

    val_hosts = ['host-a007', 'host-a008', 'host-a009']

    if version == 'trainval':
        train_scenes, val_scenes = [], []
        for s in lyft_ds.scene:
            host = '-'.join(s['name'].split('-')[:2])
            if host in val_hosts:
                val_scenes.append(s['token'])
            else:
                train_scenes.append(s['token'])
    elif version == 'test':
        train_scenes = [s['token'] for s in lyft_ds.scene]
        val_scenes = []
    else:
        raise ValueError('unknown')

    if do_test:
        print(f'test scene: {len(train_scenes)}')
    else:
        print(f'train scene: {len(train_scenes)}, val scene: {len(val_scenes)}')

    train_nusc_infos, val_nusc_infos = _fill_trainval_infos(
        lyft_ds, train_scenes, val_scenes, do_test, max_sweeps=max_sweeps)

    root_path = Path(root_path)
    metadata = {'version': version,}
    if do_test:
        print(f'test sample: {len(train_nusc_infos)}')
        data = {
            'infos': train_nusc_infos,
            'metadata': metadata,
        }
        with open(root_path / 'infos_test.pkl', 'wb') as f:
            pickle.dump(data, f)
    else:
        print(f'train sample: {len(train_nusc_infos)}, val sample: {len(val_nusc_infos)}')
        data = {
            'infos': train_nusc_infos,
            'metadata': metadata,
        }
        with open(root_path / 'infos_train.pkl', 'wb') as f:
            pickle.dump(data, f)
        data['infos'] = val_nusc_infos
        with open(root_path / 'infos_val.pkl', 'wb') as f:
            pickle.dump(data, f)


if __name__ == '__main__':
    fire.Fire()
