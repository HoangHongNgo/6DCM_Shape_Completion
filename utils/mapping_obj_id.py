import numpy as np

object_list = [-1, 0, 2, 5, 7, 8, 9, 11, 14, 15, 17,
               18, 20, 21, 22, 26, 27, 29, 30, 34, 36,
               37, 38, 40, 41, 43, 44, 46, 48, 51, 52,
               56, 57, 58, 60, 61, 62, 63, 66, 69, 70]
object_list = (np.asarray(object_list)[:] + 1).tolist()
mapping = {}
for x in range(len(object_list)):
    mapping[x] = object_list[x]


def mapping_obj_id(segMask):
    # Let object list fit mask 41 obj
    for i, cls in enumerate(mapping):
        segMask = np.where(segMask == mapping[i], i, segMask)

    return segMask
