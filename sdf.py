import numpy as np
from scipy.ndimage import distance_transform_edt


def compute_sdf(voxel_matrix):
    inside_dist = distance_transform_edt(voxel_matrix)
    outside_dist = distance_transform_edt(1 - voxel_matrix)
    sdf = outside_dist - inside_dist
    sdf = sdf / 32.0
    sdf = sdf.clip(-0.2, 0.2)
    return sdf
