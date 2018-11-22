import scipy.ndimage.interpolation as ndimage
import numpy as np
import math
import matplotlib.pyplot as plt
import os
import  shutil

def rotate_point(size, pos, rad_alpha):
    brack_area = rad_alpha%(math.pi/2)
    scale = math.cos(brack_area)*(1+math.tan(brack_area))
    matrice = [[math.cos(-rad_alpha), math.sin(-rad_alpha)], [-math.sin(-rad_alpha), math.cos(-rad_alpha)]]
    rel_pos = [pos[0]-size/2, pos[1]-size/2]
    new_pos = np.dot(rel_pos, matrice)
    result = [int(new_pos[0] + size*scale/2), int(size*scale/2 - new_pos[1])]
    return result

def crop_from_array(array, pos, rad_alpha, window_size, margin = 0):
    deg_alpha = (rad_alpha/(2*math.pi))*360
    img2 = ndimage.rotate(array, 360 - deg_alpha, cval=255)
    pos_rotated = rotate_point(array.shape[0], pos, (rad_alpha))
    result = img2[(pos_rotated[1]-int(window_size/2)):(pos_rotated[1]+int(window_size/2)), (pos_rotated[0]-int(window_size/2)+margin):(pos_rotated[0]+int(window_size/2) + margin)]
    return result

def get_dists(view, pos, max_dist, resolotion, theta_array):
    answers = []
    points = []
    step = int(max_dist/resolotion)
    for theta in theta_array:
        new_pos = [view.shape[0] - pos[0], pos[1]]
        point = view[int(new_pos[0]), int(new_pos[1])]
        for iter in range(resolotion):
            if point < 100:
                break
            new_pos = [new_pos[0] + step*math.cos(theta), new_pos[1] - step*math.sin(theta)]
            if int(new_pos[0]) > view.shape[0]:
                new_pos[0] = view.shape[0]-1
            if int(new_pos[1]) > view.shape[1]:
                new_pos[1] = view.shape[1]-1
            point = view[int(new_pos[0]), int(new_pos[1])]


        points.append([new_pos[0], new_pos[1]])
        answers.append(iter/resolotion)
    return answers, points

class Scene():
    def __init__(self, field, positions, directions, velocities):
        self.field = field
        self.positions = positions
        self.directions = directions
        self.velocities = velocities

class Map():
    def __init__(self, drawing = np.zeros([500,500,3]), starting_poses = [], starting_thetas = []):
        self.drawing = drawing
        self.starting_poses = starting_poses
        self.starting_thetas = starting_thetas

# def copy_rename(from_, name_from, to_, name_to):
#     src_file = os.path.join(from_, name_from)
#     shutil.copy(src_file,to_)
#
#     dst_file = os.path.join(to_, name_from)
#     new_dst_file_name = os.path.join(to_, name_to)
#     os.rename(dst_file, new_dst_file_name)
