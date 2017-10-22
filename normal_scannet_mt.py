import numpy as np
import cv2
import PIL
import PIL.Image as Image
import os
import time
from time import gmtime, strftime
import math
from numba import jit, double
import threading

num_thread = 8
depth_path = './data2/fr1_desk/'
normal_path = './data2/fr1_desk/normal2/'
os.makedirs(normal_path, exist_ok=True)

print('Depth folder: {}'.format(depth_path))
print('Normal folder: {}'.format(normal_path))
#if (input('Start or not ([Y]/N):') or 'Y') != 'Y':
#    exit()

@jit
def cross(vec1, vec2):
    """ Calculate the cross product of two 3d vectors. """
    result = np.zeros(3)
    return cross_(vec1, vec2, result)

@jit(nopython=True)
def cross_(vec1, vec2, result):
    """ Calculate the cross product of two 3d vectors. """
    a1, a2, a3 = double(vec1[0]), double(vec1[1]), double(vec1[2])
    b1, b2, b3 = double(vec2[0]), double(vec2[1]), double(vec2[2])
    result[0] = a2 * b3 - a3 * b2
    result[1] = a3 * b1 - a1 * b3
    result[2] = a1 * b2 - a2 * b1
    return result

@jit(nopython=True)
def norm(vec):
    """ Calculate the norm of a 3d vector. """
    return math.sqrt(vec[0]*vec[0] + vec[1]*vec[1] + vec[2]*vec[2])

@jit()
def normalize(vec):
    """ Calculate the normalized vector (norm: one). """
    result = np.copy(vec)
    return normalize1(result)

@jit(nopython = True)
def normalize1(vec):
    norm_=norm(vec)
    if norm_ == 0:
        vec[0] = 0.
        vec[1] = 0.
        vec[2] = 0.
    else:
        vec[0] = vec[0] / norm_
        vec[1] = vec[1] / norm_
        vec[2] = vec[2] / norm_
    return vec

@jit
def estimate_normal(depth_pixels):
    width = depth_pixels.shape[1]
    height = depth_pixels.shape[0]
    norm_pix_arr = np.zeros((height,width,3), 'uint8')
    for y in range(1, height-1):
        for x in range(1, width-1):
            if depth_pixels[y][x] == 0:
                continue
            """
            * * t * *
            * l c * *
            * * * * *
            """
            vec_t = np.array([x, y-1, depth_pixels[y-1][x]])
            vec_l = np.array([x-1, y, depth_pixels[y][x-1]])
            vec_c = np.array([x, y, depth_pixels[y][x]])
            vec_nd = normalize(cross(vec_l - vec_c, vec_t - vec_c))
            vec_nd = (vec_nd + 1.) / 2.0 * 255
            norm_pix_arr[y][x] = vec_nd
    return norm_pix_arr



class myThread(threading.Thread):
    def __init__(self, thread_id, name, counter, filelist, startf, endf, img_width, img_height):
        threading.Thread.__init__(self)
        self.threadID = thread_id
        self.name = name
        self.counter = counter
        self.files = filelist
        self.startf = startf
        self.endf = endf
        self.img_width = img_width
        self.img_height = img_height

    def run(self):
        for file in self.files[self.startf:self.endf]:
            if not os.path.isdir(file) and file.find('.png') != -1:
                depth_image = np.asarray(Image.open(os.path.join(depth_path, file))).astype('float32')
                depth_image = cv2.bilateralFilter(depth_image, 8, 40, 40)
                start = time.time()
                norm_pix_arr = estimate_normal(depth_image)
                print('Thread {}: Frame {} -- Time: {:.3f}'.format(self.threadID, file, time.time()-start))
                normal_image = Image.fromarray(norm_pix_arr)
                depth_filename = os.path.join(normal_path, file)
                normal_image.save(depth_filename)



width = 640
height = 480
files = os.listdir(depth_path)
files.sort()
for file in files:
    if not os.path.isdir(file) and file.find('.png') != -1:
        img = Image.open(os.path.join(depth_path, file))
        width = img.width
        height = img.height
        break

# Create new threads
thread_list = []
for tid in range(num_thread):
    sf = int(tid / num_thread * len(files))
    ef = int((tid+1) / num_thread * len(files))
    thread_list.append(myThread(tid, 't{}'.format(tid), tid, files, sf, ef, width, height))

# Start threads
for thread in thread_list:
    thread.start()

# Join threads
for thread in thread_list:
    thread.join()

print('All threads are done.')