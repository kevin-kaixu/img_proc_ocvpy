import numpy as np
from scipy import ndimage
from PIL import Image
import PIL.ImageOps as iop
import os


coplanarity_path = './data2/fr1_desk/segmentation_images/coplanarity.txt'
seg_path = './data2/fr1_desk/segmentation_images/'

frame_dict = {}
copi_list = []
pos_pair_list = []
neg_pair_list = []

class CopInfo:
    def __init__(self, frame1, frame2, seg1, seg2, vdist, vangle, sdist, s1p2dist, s2p1dist, nangle, affinity):
        self.f1 = frame1        # file name of frame 1
        self.f2 = frame2        # file name of frame 2
        self.s1 = seg1          # id of segment 1
        self.s2 = seg2          # id of segment 2
        self.vd = vdist         # distance between the image viewpoints
        self.va = vangle        # angle (in radians) between the image view directions
        self.sd = sdist         # distance between the segment centroids
        self.s1p2 = s1p2dist    # distance from the centroid of segment 1 to the plane of segment 2
        self.s2p1 = s2p1dist    # distance from the centroid of segment 1 to the plane of segment 2
        self.na = nangle        # angle (in radians) between the segment normals
        self.a = affinity       # heuristic score combining the last four values (1 is best, 0 is worst).
                                #   It is used to determine which pairs to output (affinity > 0.1)
        self.h = 0.             # hardness: 0.0 means easiest and 1.o means hardest

    def eval_hardness(self, maxvd, maxva, maxsd):
        self.h = 1. - (1. - self.vd / maxvd) * (1. - self.va / maxva) * (1. - self.sd / maxsd)


def GetATriplet():
    
    return

def GetFrameList():
    files = os.listdir(seg_path)
    files.sort()
    for fn in files:
        if not os.path.isdir(fn):
            if fn.find('txt') != -1:
                lines = open(fn)
                num_seg = len(lines)
                frame_dict[''.join(fn[0:fn.find('_')].split('.'))] = num_seg
    return frame_dict

with open(coplanarity_path) as cpf:
    for line in cpf:
        kw = line.split()
        copi = CopInfo(frame1=kw[0], frame2=kw[1], seg1=int(kw[2]), seg2=int(kw[3]),
                       vdist=float(kw[4]), vangle=float(kw[5]), sdist=float(kw[6]),
                       s1p2dist=float(kw[7]), s2p1dist=float(kw[8]), nangle=float(kw[9]), affinity=float(kw[10]))
        pos_pair_list.append(sorted([int(''.join(kw[0].split('.'))+kw[2]), int(''.join(kw[1].split('.'))+kw[3])]))
        copi_list.append(copi)

max_vdist = max(copi.vd for copi in copi_list)
min_vdist = min(copi.vd for copi in copi_list)
max_vangle = max(copi.va for copi in copi_list)
min_vangle = min(copi.va for copi in copi_list)
max_sdist = max(copi.sd for copi in copi_list)
min_sdist = min(copi.sd for copi in copi_list)
max_spdist = max(max(copi.s1p2 for copi in copi_list), max(copi.s2p1 for copi in copi_list))
min_spdist = min(min(copi.s1p2 for copi in copi_list), min(copi.s2p1 for copi in copi_list))
max_nangle = max(copi.na for copi in copi_list)
min_nangle = min(copi.na for copi in copi_list)
max_affinity = max(copi.a for copi in copi_list)
min_affinity = min(copi.a for copi in copi_list)
print('max_vdist = ', max_vdist)
print('min_vdist = ', min_vdist)
print('max_vangle = ', max_vangle)
print('min_vangle = ', min_vangle)
print('max_sdist = ', max_sdist)
print('min_sdist = ', min_sdist)
print('max_spdist = ', max_spdist)
print('min_spdist = ', min_spdist)
print('max_nangle = ', max_nangle)
print('min_nangle = ', min_nangle)
print('max_affinity = ', max_affinity)
print('min_affinity = ', min_affinity)

# Evaluate hardness
for copi in copi_list:
    #copi.eval_hardness(max_vdist, max_vangle, max_sdist)
    copi.eval_hardness(2.5, 1.8, 3.5)

max_hardness = max(copi.h for copi in copi_list)
min_hardness = min(copi.h for copi in copi_list)
print('max_hardness = ', max_hardness)
print('min_hardness = ', min_hardness)

GetFrameList()