import os
import glob
import numpy as np
import time
import torch
from torch.utils import data

from torch.utils.data import DataLoader
from utils import mapping_obj_id, mapping, object_list
from utils import HParam
from reader import read_comap_np, read_depth_np, read_diff, read_mask_np, read_rgb_np, ssd2pointcloud, create_point_cloud_from_depth_image
from resizer import resizer


class SCM_Dataset(data.Dataset):
    def __init__(self, gn_root, camera, split='train', pred_depth=False, pred_cloud=False, pred_pcn=False):
        self.gn_root = gn_root
        self.camera = camera
        self.split = split
        self.pred_depth = pred_depth
        self.pred_cloud = pred_cloud
        self.pred_pcn = pred_pcn

        self.rgbList = []
        self.depthList = []
        self.comapfList = []
        self.comapbList = []
        self.diffList = []
        self.segMaskList = []
        self.kList = []

        if self.split == 'train':
            data_set = range(0, 100)
        elif self.split == 'valid':
            data_set = range(100, 130, 3)
        elif self.split == 'test':
            data_set = range(100, 130)

        for sceneId in data_set:
            try:

                self.rgbList += sorted(glob.glob(os.path.join(self.gn_root, 'scenes',
                                       'scene_'+str(sceneId).zfill(4), self.camera, 'rgb', '*.png')))
                self.depthList += sorted(glob.glob(os.path.join(self.gn_root, "scenes",
                                         "scene_"+str(sceneId).zfill(4), self.camera, 'depth', '*.png')))
                self.comapfList += sorted(glob.glob(os.path.join(
                    self.gn_root, 'comap', 'scene_'+str(sceneId).zfill(4), self.camera, 'cmpf', '*.png')))
                self.comapbList += sorted(glob.glob(os.path.join(self.gn_root, 'comap',
                                          'scene_' + str(sceneId).zfill(4), self.camera, 'cmpb', '*.png')))
                self.diffList += sorted(glob.glob(os.path.join(self.gn_root, 'comap',
                                        'scene_' + str(sceneId).zfill(4), self.camera, 'diff', '*.npz')))
                self.segMaskList += sorted(glob.glob(os.path.join(
                    self.gn_root, 'scenes', 'scene_'+str(sceneId).zfill(4), self.camera, 'label', '*.png')))

                for frameId in range(0, 256):
                    self.kList.append(os.path.join(
                        self.gn_root, 'scenes', 'scene_' + str(sceneId).zfill(4), self.camera, 'camK.npy'))
            except:
                continue

    def create_rgbd(self, rgb, depth):
        rgbd = np.append(rgb, np.expand_dims(depth, axis=2), axis=2)
        return rgbd

    def create_groundtruth(self, comap, segMask, diff, cloud):
        if self.pred_depth:
            map = np.expand_dims(diff, axis=2)
        elif self.pred_cloud:
            map = cloud
        else:
            map = comap

        segMask = mapping_obj_id(segMask)

        gt = np.append(np.expand_dims(segMask, axis=2), map, axis=2)
        return gt

    # TODO function to create point cloud rear and front

    def one_hot_encoding(self, segMask):
        h, w = segMask.shape
        one_hot_encode = np.zeros((h, w, len(object_list)))
        for i, cls in enumerate(mapping):
            one_hot_encode[:, :, i] = np.asarray(segMask == mapping[i])
        return one_hot_encode

    def one_hot_decoding(self, one_hot):
        segMask = np.zeros(one_hot.shape[:2])
        single_layer = np.argmax(one_hot, axis=-1)
        for k in mapping.keys():
            segMask[single_layer == k] = mapping[k]
        segMask = np.asarray(segMask, dtype='int')
        return segMask

    def get_onject_list(self):
        return object_list

    def __getraw__(self, index):
        rgb = read_rgb_np(self.rgbList[index])
        depth = read_depth_np(self.depthList[index])
        comap = read_comap_np(self.comapfList[index], self.comapbList[index])
        segMask = read_mask_np(self.segMaskList[index])
        diff = read_diff(self.diffList[index])
        k = np.load(self.kList[index])
        return rgb, depth, comap, segMask, diff, k

    def __getitem__(self, index_tuple):
        if self.split == 'train':
            index, height, width = index_tuple
        else:
            index = index_tuple

        rgb, depth, comap, segMask, diff, k = self.__getraw__(index)
        rgbd = self.create_rgbd(rgb, depth)

        # Front point cloud
        scene_cloud = create_point_cloud_from_depth_image(depth, k)
        # GT rear point cloud
        rear_cloud = ssd2pointcloud(scene_cloud, segMask, diff)

        # Downsample resolution
        if self.split == 'train' and self.pred_cloud == False:
            rgbd, segMask, comap, rear_cloud, diff = resizer(
                rgbd, segMask, comap, rear_cloud, diff)
        elif self.split == 'valid':
            rgbd, segMask, comap, rear_cloud, diff = resizer(
                rgbd, segMask, comap, rear_cloud, diff)

        gt = self.create_groundtruth(comap, segMask, diff, rear_cloud)

        input_tensor = torch.from_numpy(rgbd).permute(2, 0, 1).float()

        gt_tensor = torch.from_numpy(gt).permute(2, 0, 1).float()

        fr_cloud_tensor = torch.from_numpy(
            scene_cloud).permute(2, 0, 1).float()

        if self.pred_cloud:
            # TODO data = {rgbd, cloud}
            pass
        elif self.pred_depth:
            # TODO data = {rgbd, depth}
            pass
        elif self.pred_pcd:
            # TODO data = {rgbd, pcd}
            pass

        data = {"rgbd": input_tensor, "ground_truth": gt_tensor}
        return data

    def __len__(self):
        return len(self.rgbList)


if __name__ == '__main__':
    t0 = time.time()
    gn_root = "/media/dsp520/Grasp_2T/graspnet"
    config = '/media/dsp520/Grasp_2T/6DCM_Grasp/6DCM/configs/default.yaml'
    camera = 'realsense'

    hp = HParam(config)
    dataset = SCM_Dataset(gn_root, camera, split="test", pred_depth=True)
    val_dataloader = DataLoader(
        dataset, batch_size=1, num_workers=0, shuffle=False)
    t1 = time.time()
    print('start')

    for idx, data in enumerate(val_dataloader):
        # t1 = time.time()
        input = data['rgbd']
        label = data['gt']  # (b, 4, h, w)
        # print(idx)
    t2 = time.time()
    print(t2 - t1)
