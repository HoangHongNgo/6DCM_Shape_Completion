import cv2
import numpy as np


def resizer(rgbd, mask, comap, cloud, diff):
    original_width, original_height = mask.shape

    rgb = rgbd[:, :, :3]

    rgb = cv2.resize(rgb, (int(original_height/2),
                     int(original_width/2)), interpolation=cv2.INTER_AREA)

    depth = rgbd[:, :, 3]
    depth = cv2.resize(depth, (int(original_height/2),
                       int(original_width/2)), interpolation=cv2.INTER_NEAREST)
    rgbd = np.append(rgb, np.expand_dims(depth, axis=2), axis=2)

    mask = cv2.resize(mask, (int(original_height/2),
                      int(original_width/2)), interpolation=cv2.INTER_NEAREST)
    cmpf = comap[..., :3]
    cmpb = comap[..., 3:]
    cmpf = cv2.resize(cmpf, (int(original_height/2),
                      int(original_width/2)), interpolation=cv2.INTER_AREA)
    cmpb = cv2.resize(cmpb, (int(original_height/2),
                      int(original_width/2)), interpolation=cv2.INTER_AREA)
    comap = np.append(cmpf, cmpb, axis=2)
    cloud = cv2.resize(cloud, (int(original_height/2),
                       int(original_width/2)), interpolation=cv2.INTER_NEAREST)
    diff = cv2.resize(diff, (int(original_height/2),
                      int(original_width/2)), interpolation=cv2.INTER_AREA)

    return rgbd, mask, comap, cloud, diff


def resizer_input(rgbd, mask):
    original_width, original_height = mask.shape
    rgbd = cv2.resize(rgbd, (int(original_height/2),
                      int(original_width/2)), interpolation=cv2.INTER_AREA)
    mask = cv2.resize(mask, (int(original_height/2),
                      int(original_width/2)), interpolation=cv2.INTER_NEAREST)
    return rgbd, mask
