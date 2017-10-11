import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import PIL
import PIL.Image as Image
import os

color_path = './data'
normal_path = './data/Normal'
seg_path = './data/segmentation_images'

class SegInfo:
    def __init__(self, minx, miny, maxx, maxy, width, height):
        self.minx = minx
        self.miny = miny
        self.maxx = maxx
        self.maxy = maxy
        self.width = width
        self.height = height
        self.mask = Image.new('1', size=(width,height))

def GetFileNames(color_dir, normal_dir, seg_dir):
    list_dir = os.walk(color_dir)
    files = os.listdir(color_dir)
    col_fn_list = []
    depth_fn_list = []
    for f in files:
        if not os.path.isdir(f):
            if f.find('color') != -1:
                col_fn_list.append(f)
            if f.find('depth') != -1:
                depth_fn_list.append(f)

    files = os.listdir(normal_dir)
    normal_fn_list = []
    for f in files:
        if not os.path.isdir(f):
            normal_fn_list.append(f)

    files = os.listdir(seg_dir)
    seg_fn_list = []
    for f in files:
        if not os.path.isdir(f):
            if f.find('png') != -1:
                seg_fn_list.append(f)

    return (col_fn_list, depth_fn_list, normal_fn_list, seg_fn_list)


def GetSegmentsInfo(filename):
    pixels = np.asarray(Image.open(filename))
    width = pixels.shape[1]
    height = pixels.shape[0]

    txt_fn = filename[0:len(filename)-len('.png')]+'.txt'
    seg_info_list = []
    with open(txt_fn) as tf:
        for line in tf:
            seg_info_list.append(SegInfo(minx=width, miny=height, maxx=0, maxy=0, width=width, height=height))
        
    all_seg_mask = Image.new('RGB', size=(width,height), color=(0, 0, 0))
    asm_map = all_seg_mask.load()
    num_color = 32
    cp = sns.color_palette("hls", num_color)
    cpi = [(int(x[0]*255),int(x[1]*255),int(x[2]*255)) for x in cp]

    for y in range(height):
        for x in range(width):
            seg_id = pixels[y][x] - 1
            if seg_id != -1:
                seg_info_list[seg_id].minx = min(x, seg_info_list[seg_id].minx)
                seg_info_list[seg_id].miny = min(y, seg_info_list[seg_id].miny)
                seg_info_list[seg_id].maxx = max(x, seg_info_list[seg_id].maxx)
                seg_info_list[seg_id].maxy = max(y, seg_info_list[seg_id].maxy)
                pixel_map = seg_info_list[seg_id].mask.load()
                pixel_map[x,y] = 1
                asm_map[x,y] = cpi[seg_id % num_color]
    
    return seg_info_list, all_seg_mask


def CreateACrop(img, left, upper, right, lower):
    img_xmin = 0
    img_ymin = 0
    img_xmax = img.width
    img_ymax = img.height
    
    crop_centx = left + (right - left) / 2
    crop_centy = lower + (upper - lower) / 2
    crop_size = int(max((right - left), (lower - upper)))
    crop_hsize = int(crop_size / 2)
    crop_xmin = int(crop_centx - crop_hsize)
    crop_xmax = int(crop_centx + crop_hsize)
    crop_ymin = int(crop_centy - crop_hsize)
    crop_ymax = int(crop_centy + crop_hsize)

    cross_boundary = False
    shift = [0, 0]
    if crop_xmin < img_xmin :
        shift[0] = int(img_xmin - crop_xmin)
        crop_xmin = img_xmin
        cross_boundary = True
    if crop_ymin < img_ymin :
        shift[1] = int(img_ymin - crop_ymin)
        crop_ymin = img_ymin
        cross_boundary = True
    if crop_xmax > img_xmax :
        crop_xmax = img_xmax
        cross_boundary = True
    if crop_ymax > img_ymax :
        crop_ymax = img_ymax
        cross_boundary = True

    crop_img = img.crop((crop_xmin, crop_ymin, crop_xmax, crop_ymax))
    if not cross_boundary :
        return crop_img
    else:
        new_img = Image.new('RGB', size=(crop_size,crop_size), color=(128, 128, 128))
        new_img.paste(crop_img, tuple(shift))
        return new_img

filenames_list = GetFileNames(color_path, normal_path, seg_path)

seg_info_list,all_seg_mask = GetSegmentsInfo(seg_path+'/'+filenames_list[3][0])

img_fn = color_path+'/'+filenames_list[0][0]
img = Image.open(img_fn)

img_mask = Image.blend(img, all_seg_mask, alpha=0.7)
img_mask_fn = img_fn[0:len(img_fn)-len('.jpg')]+'_mask.jpg'
img_mask.save(img_mask_fn)

for seg, idx in zip(seg_info_list, range(len(seg_info_list))):
    if seg.minx == seg.maxx or seg.miny == seg.maxy:
        continue
    img_crop = CreateACrop(img, seg.minx, seg.miny, seg.maxx, seg.maxy)
    # img_crop = img.crop((seg.minx, seg.miny, seg.maxx, seg.maxy))
    img_crop_fn = img_fn[0:len(img_fn)-len('.jpg')]+'_seg_{}'.format(idx)+'.jpg'
    img_crop.save(img_crop_fn)
    img_mask_crop = CreateACrop(seg.mask, seg.minx, seg.miny, seg.maxx, seg.maxy)
    # img_mask_crop = seg.mask.crop((seg.minx, seg.miny, seg.maxx, seg.maxy))
    img_mask_crop_fn = img_fn[0:len(img_fn)-len('.jpg')]+'_seg_{}_mask'.format(idx)+'.jpg'
    img_mask_crop.save(img_mask_crop_fn)


