"""
support ScanNet dataset
support multi-thread
no support to jumping segmentation file sequence (num. of color images is not equal to that of segmentation info.)
"""

import numpy as np
import seaborn as sns
from scipy import ndimage
import matplotlib.pyplot as plt
import PIL
import PIL.Image as Image
import PIL.ImageOps as ImgOps
from PIL import ImageChops
import os
import time
from time import gmtime, strftime
import util
from rgbd_cal import rgbd_cal_dmask2cbox
import threading


class SegInfo:
    def __init__(self, minx=0, miny=0, maxx=0, maxy=0, width=1, height=1):
        self.minx = minx
        self.miny = miny
        self.maxx = maxx
        self.maxy = maxy
        self.numpnt = 0
        self.mask = Image.new('1', size=(width, height))


def GetFileNames(color_dir, depth_dir, normal_dir, seg_dir):
    files = os.listdir(color_dir)
    files.sort()
    col_fn_list = []
    for f in files:
        if not os.path.isdir(f):
            if f.find('.jpg') != -1:
                col_fn_list.append(f)
    
    files = os.listdir(depth_dir)
    files.sort()
    depth_fn_list = []
    for f in files:
        if not os.path.isdir(f):
            if f.find('.png') != -1:
                depth_fn_list.append(f)

    files = os.listdir(normal_dir)
    files.sort()
    normal_fn_list = []
    for f in files:
        if not os.path.isdir(f):
            if f.find('.png') != -1:
                normal_fn_list.append(f)

    files = os.listdir(seg_dir)
    files.sort()
    seg_fn_list = []
    for f in files:
        if not os.path.isdir(f):
            if f.find('.png') != -1:
                seg_fn_list.append(f)

    return (col_fn_list, depth_fn_list, normal_fn_list, seg_fn_list)


def GetSegmentsInfo_ff(filename):
    # Open the segmentation image containing the labels for all segments (one label per pixel)
    seg_img = Image.open(filename)
    seg_img = seg_img.convert('L')
    # Open the txt file to see how many segments are there in the segmentation
    txt_fn = filename[0:len(filename)-len('.png')]+'.txt'
    num_seg = sum(1 for line in open(txt_fn))   # the num. of segments is equal to that of lines
    seg_info_list = [SegInfo() for i in range(num_seg)]
    # Create a subtracter image (all 1) used to peel a segment
    subtracter_img = Image.new('L', size=seg_img.size, color=1)
    # Create a background mask: 1 for foreground (segments) and 0 for background (non-segments)
    background_mask = seg_img.point(lambda p: p > 0 and 255).convert('1', dither=Image.NONE)
    # Loop over all segments
    for idx in range(num_seg):
        # Peel one segment by subtract the all-1 subtracter image
        seg_img = ImageChops.subtract(seg_img, subtracter_img)
        # Binarize the segmentation image with the current segment peeled
        bi_seg_img = seg_img.point(lambda p: p > 0 and 255)
        # Invert the peeled, binarized segmentation image
        bool_img = ImgOps.invert(bi_seg_img).convert('1', dither=Image.NONE)
        # Logical AND between the peeled segmentation and background mask gets the segment image (1 for the segment)
        bool_img = ImageChops.logical_and(bool_img, background_mask)
        # Obtain the bounding box of the segment region (region with non-zero pixels)
        bbox = bool_img.getbbox()
        if bbox:
            seg_info_list[idx].minx = bbox[0]
            seg_info_list[idx].miny = bbox[1]
            seg_info_list[idx].maxx = bbox[2]
            seg_info_list[idx].maxy = bbox[3]
            # Count the number of non-zero pixels in that region
            seg_info_list[idx].numpnt = sum(bool_img.crop(bbox).point(bool).getdata())
            seg_info_list[idx].mask = bool_img
        # Update the background mask for the next loop (subtract the current segment)
        background_mask = bi_seg_img.convert('1')
    return seg_info_list


def GetSegmentsInfo_fast(filename):
    # Open the segmentation image containing the labels for all segments (one label per pixel)
    seg_img = Image.open(filename)
    seg_img = seg_img.convert('L')
    # Open the txt file to see how many segments are there in the segmentation
    txt_fn = filename[0:len(filename)-len('.png')]+'.txt'
    num_seg = sum(1 for line in open(txt_fn))   # the num. of segments is equal to that of lines
    seg_info_list = [SegInfo() for i in range(num_seg)]
    # Open the colored all segment mask image
    all_seg_mask = seg_img.convert('RGB')
    cp = sns.color_palette("hls", config.num_col)   # color palette
    cpi = [(int(x[0]*255),int(x[1]*255),int(x[2]*255)) for x in cp]
    # Create a subtracter image (all 1) used to peel a segment
    subtracter_img = Image.new('L', size=seg_img.size, color=1)
    # Create a background mask: 1 for foreground (segments) and 0 for background (non-segments)
    background_mask = seg_img.point(lambda p: p > 0 and 255).convert('1', dither=Image.NONE)
    # Loop over all segments
    for idx in range(num_seg):
        # Peel one segment by subtract the all-1 subtracter image
        seg_img = ImageChops.subtract(seg_img, subtracter_img)
        # Binarize the segmentation image with the current segment peeled
        bi_seg_img = seg_img.point(lambda p: p > 0 and 255)
        # Invert the peeled, binarized segmentation image
        bool_img = ImgOps.invert(bi_seg_img).convert('1', dither=Image.NONE)
        # Logical AND between the peeled segmentation and background mask gets the segment image (1 for the segment)
        bool_img = ImageChops.logical_and(bool_img, background_mask)
        # Obtain the bounding box of the segment region (region with non-zero pixels)
        bbox = bool_img.getbbox()
        if bbox:
            seg_info_list[idx].minx = bbox[0]
            seg_info_list[idx].miny = bbox[1]
            seg_info_list[idx].maxx = bbox[2]
            seg_info_list[idx].maxy = bbox[3]
            # Count the number of non-zero pixels in that region
            seg_info_list[idx].numpnt = sum(bool_img.crop(bbox).point(bool).getdata())
            seg_info_list[idx].mask = bool_img
        # Update the background mask for the next loop (subtract the current segment)
        background_mask = bi_seg_img.convert('1')
        all_seg_mask.paste(Image.new('RGB', all_seg_mask.size, (cpi[idx % config.num_col])), bool_img)
    return seg_info_list, all_seg_mask


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
    cp = sns.color_palette("hls", config.num_col)
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
                pixel_map[x,y] = True
                seg_info_list[seg_id].numpnt += 1
                asm_map[x,y] = cpi[seg_id % config.num_col]
    
    return seg_info_list, all_seg_mask


def CreateContextMask(img):
    img = img.convert('L')
    inv_img = ImgOps.invert(img)
    inv_arr = np.array(inv_img)
    dt = ndimage.distance_transform_edt(inv_arr)
    tmax = np.amax(dt)
    scl = 255./tmax
    dt *= scl
    dt_img = Image.fromarray(dt.astype('uint8'), 'L')
    dt_img = ImgOps.invert(dt_img)
    src = Image.new('L',size=img.size)
    dt_img.paste(im=src, mask=img)
    return dt_img


def CreateCroppedImage(img, left, upper, right, lower, fill_col, scl_fac=1, truncate=False):
    img_xmin = 0
    img_ymin = 0
    img_xmax = img.width-1
    img_ymax = img.height-1

    crop_centx = left + (right - left) / 2
    crop_centy = lower + (upper - lower) / 2
    crop_inner_size = round(max((right - left), (lower - upper)))
    crop_size = crop_inner_size * scl_fac
    crop_hsize = round(crop_size / 2)
    crop_xmin = round(crop_centx - crop_hsize)
    crop_xmax = round(crop_centx + crop_hsize)
    crop_ymin = round(crop_centy - crop_hsize)
    crop_ymax = round(crop_centy + crop_hsize)

    cross_boundary = False
    cross_x = cross_y = 0
    if crop_xmin < img_xmin:
        crop_xmin = img_xmin
        cross_boundary = True
        cross_x = -1
    if crop_ymin < img_ymin:
        crop_ymin = img_ymin
        cross_boundary = True
        cross_y = -1
    if crop_xmax > img_xmax:
        crop_xmax = img_xmax
        cross_boundary = True
        cross_x = 1
    if crop_ymax > img_ymax:
        crop_ymax = img_ymax
        cross_boundary = True
        cross_y = 1

    if not cross_boundary:
        return img.crop((crop_xmin, crop_ymin, crop_xmax, crop_ymax))
    else:
        shift = [0, 0]
        if truncate:
            # The region beyong the boundary will be truncated, since we now
            #   compute a new crop size based on the truncated extents
            crop_size_x = crop_xmax - crop_xmin
            crop_size_y = crop_ymax - crop_ymin
            if crop_size_x < crop_size_y and cross_x == 1:
                shift[0] = crop_size_y - crop_size_x
            if crop_size_x > crop_size_y and cross_y == 1:
                shift[1] = crop_size_x - crop_size_y
            crop_size = max(crop_size_x, crop_size_y)
        else:
            # The region beyong the boundary is not truncated by filling with grey
            #   since we are using the desired crop size
            if cross_x == -1:
                shift[0] = img_xmin - crop_xmin
            if cross_y == -1:
                shift[1] = img_ymin - crop_ymin
        new_img = Image.new(img.mode, (crop_size, crop_size), fill_col)
        new_img.paste(img.crop((crop_xmin, crop_ymin, crop_xmax, crop_ymax)), tuple(shift))
        return new_img

def isExist(result_path, col_img_fn, depth_img_fn, normal_img_fn, seg_id):
    fn = os.path.join(result_path, col_img_fn[0:len(col_img_fn)-len('.color.jpg')]+'_seg_{}_l_col'.format(seg_id)+'.png')
    fn2 = os.path.join(result_path, col_img_fn[0:len(col_img_fn)-len('.color.jpg')]+'_seg_{}_l_color'.format(seg_id)+'.png')
    if os.path.exists(fn):
        os.rename(fn, fn2)
    elif not os.path.exists(fn2):
        return False
    fn = os.path.join(result_path, col_img_fn[0:len(col_img_fn)-len('.color.jpg')]+'_seg_{}_g_col'.format(seg_id)+'.png')
    fn2 = os.path.join(result_path, col_img_fn[0:len(col_img_fn)-len('.color.jpg')]+'_seg_{}_g_color'.format(seg_id)+'.png')
    if os.path.exists(fn):
        os.rename(fn, fn2)
    elif not os.path.exists(fn2):
        return False
    fn = os.path.join(result_path, depth_img_fn[0:len(depth_img_fn)-len('.depth.pgm')]+'_seg_{}_l_depth'.format(seg_id)+'.pgm')
    if not os.path.exists(fn):
        return False
    fn = os.path.join(result_path, depth_img_fn[0:len(depth_img_fn)-len('.depth.pgm')]+'_seg_{}_g_depth'.format(seg_id)+'.pgm')
    if not os.path.exists(fn):
        return False
    fn = os.path.join(result_path, normal_img_fn[0:len(normal_img_fn)-len('.depth.jpg')]+'_seg_{}_l_normal'.format(seg_id)+'.png')
    if not os.path.exists(fn):
        return False
    fn = os.path.join(result_path, normal_img_fn[0:len(normal_img_fn)-len('.depth.jpg')]+'_seg_{}_g_normal'.format(seg_id)+'.png')
    if not os.path.exists(fn):
        return False
    fn = os.path.join(result_path, col_img_fn[0:len(col_img_fn)-len('.color.jpg')]+'_seg_{}_l_mask'.format(seg_id)+'.png')
    if not os.path.exists(fn):
        return False
    fn = os.path.join(result_path, col_img_fn[0:len(col_img_fn)-len('.color.jpg')]+'_seg_{}_g_mask'.format(seg_id)+'.png')
    if not os.path.exists(fn):
        return False
    fn = os.path.join(result_path, col_img_fn[0:len(col_img_fn)-len('.color.jpg')]+'_seg_{}_g_mask_c'.format(seg_id)+'.png')
    if not os.path.exists(fn):
        return False
    return True


config = util.get_args()
if not config.no_config_file:
    print('Using config file: ', config.config_file)
    config = util.get_config_from_file(config.config_file, config)
else:
    print('No config file is prescribed')

color_path = config.color_path
depth_path = config.depth_path
normal_path = config.normal_path
seg_path = config.seg_path
camera_path = config.camera_path
result_path = config.result_path
result_mask_path = config.result_mask_path

rgbd_calibration = False
color_intr_path = os.path.join(camera_path, 'COLOR_INTRINSICS')
if os.path.isfile(color_intr_path):
    color_intr = np.loadtxt(color_intr_path, usecols=range(3))
    print('Using RGB camera intrinsics.')
color_extr_path = os.path.join(camera_path, 'COLOR_EXTRINSICS')
if os.path.isfile(color_extr_path):
    color_extr = np.loadtxt(color_extr_path, usecols=range(4))
    print('Using RGB camera extrinsics.')
depth_intr_path = os.path.join(camera_path, 'DEPTH_INTRINSICS')
if os.path.isfile(depth_intr_path):
    depth_intr = np.loadtxt(depth_intr_path, usecols=range(3))
    print('Using depth camera intrinsics.')
depth_extr_path = os.path.join(camera_path, 'DEPTH_EXTRINSICS')
if os.path.isfile(depth_extr_path):
    depth_extr = np.loadtxt(depth_extr_path, usecols=range(4))
    print('Using depth camera extrinsics.')
if os.path.isfile(color_intr_path) and os.path.isfile(color_extr_path) and os.path.isfile(depth_intr_path) and os.path.isfile(depth_extr_path):
    rgbd_calibration = True

filenames_list = GetFileNames(color_path, depth_path, normal_path, seg_path)
os.makedirs(result_path, exist_ok=True)
os.makedirs(result_mask_path, exist_ok=True)

grey_color = (128, 128, 128)
black_color = (0, 0, 0)

num_frame = len(filenames_list[0])
print('Frame file information:')
print('# Color image (frame): ', len(filenames_list[0]))
print('   >>> [{}] --> [{}]'.format(filenames_list[0][0], filenames_list[0][-1]))
print('# Depth map: ', len(filenames_list[1]))
print('   >>> [{}] --> [{}]'.format(filenames_list[1][0], filenames_list[1][-1]))
print('# Normal map: ', len(filenames_list[2]))
print('   >>> [{}] --> [{}]'.format(filenames_list[2][0], filenames_list[2][-1]))
print('# Segmentation info.: ', len(filenames_list[3]))
print('   >>> [{}] --> [{}]'.format(filenames_list[3][0], filenames_list[3][-1]))
if (input('Start or not ([Y]/N):') or 'Y') != 'Y':
    exit()
print('Starting generation ...')
start = time.time()


class myThread(threading.Thread):
    def __init__(self, thread_id, name, counter, startf, endf):
        threading.Thread.__init__(self)
        self.threadID = thread_id
        self.name = name
        self.counter = counter
        self.startf = startf
        self.endf = endf

    def run(self):
        for fr in range(self.startf, self.endf):
            # Get segmentation information
            seg_info_list, all_seg_mask = GetSegmentsInfo_fast(os.path.join(seg_path, filenames_list[3][fr]))
            #seg_info_list = GetSegmentsInfo_ff(os.path.join(seg_path, filenames_list[3][fr]))
            # Open color image
            col_img_fn = filenames_list[0][fr]
            col_img = Image.open(os.path.join(color_path, col_img_fn))
            # Generate segmentation masked color image
            #img_mask = Image.blend(col_img, all_seg_mask, alpha=config.blend_alpha)
            #img_mask_fn = col_img_fn[0:len(col_img_fn)-len('.jpg')]+'_mask.jpg'
            #img_mask.save(os.path.join(result_mask_path, img_mask_fn))
            # Open depth image
            depth_img_fn = filenames_list[1][fr]
            depth_img = Image.open(os.path.join(depth_path, depth_img_fn))
            # Open normal map
            normal_img_fn = filenames_list[2][fr]
            normal_img = Image.open(os.path.join(normal_path, normal_img_fn))
            # Print some statistics
            if not config.no_print and fr % config.print_every == 0:
                print('Thread {}: Progress - {:>6.0f} frame / {} frame'.format(self.threadID, fr, num_frame))
            # For each segment, generate cropped color image, depth image, normal map and segment mask
            for seg, seg_id in zip(seg_info_list, range(len(seg_info_list))):
                if seg.numpnt < config.tol_size:
                    print('Thread {}:   >>> Segment {} of frame {} is being ignored (size < {}).'.format(self.threadID, seg_id, fr, config.tol_size))
                    continue
                if isExist(result_path, col_img_fn, depth_img_fn, normal_img_fn, seg_id):
                    print('Thread {}:   >>> Segment {} of frame {} exists.'.format(self.threadID, seg_id, fr))
                    continue
                # Crop color image
                if rgbd_calibration:
                    rgb_bbox = rgbd_cal_dmask2cbox(np.asarray(depth_img), col_img.height, col_img.width, depth_intr, color_intr, depth_extr, 1000, np.asarray(seg.mask))
                    img_crop = CreateCroppedImage(col_img, rgb_bbox[0], rgb_bbox[1], rgb_bbox[2], rgb_bbox[3], fill_col=grey_color)   # local crop
                else:
                    img_crop = CreateCroppedImage(col_img, seg.minx, seg.miny, seg.maxx, seg.maxy, fill_col=grey_color)   # local crop
                img_crop = img_crop.resize((224,224), resample=PIL.Image.BICUBIC)
                img_crop_fn = col_img_fn[0:len(col_img_fn)-len('.jpg')]+'_seg_{}_l_color'.format(seg_id)+'.png'
                img_crop.save(os.path.join(result_path, img_crop_fn))
                if rgbd_calibration:
                    img_crop = CreateCroppedImage(col_img, rgb_bbox[0], rgb_bbox[1], rgb_bbox[2], rgb_bbox[3], fill_col=grey_color, scl_fac=5, truncate=True)   # local crop
                else:
                    img_crop = CreateCroppedImage(col_img, seg.minx, seg.miny, seg.maxx, seg.maxy, fill_col=grey_color, scl_fac=5, truncate=True)    # global crop
                img_crop = img_crop.resize((224,224), resample=PIL.Image.BICUBIC)
                img_crop_fn = col_img_fn[0:len(col_img_fn)-len('.jpg')]+'_seg_{}_g_color'.format(seg_id)+'.png'
                img_crop.save(os.path.join(result_path, img_crop_fn))
                # Crop depth image
                img_crop = CreateCroppedImage(depth_img, seg.minx, seg.miny, seg.maxx, seg.maxy, fill_col=0)   # local crop
                img_crop = img_crop.resize((224,224), resample=PIL.Image.BICUBIC)
                img_crop_fn = depth_img_fn[0:len(depth_img_fn)-len('.png')]+'_seg_{}_l_depth'.format(seg_id)+'.pgm'
                img_crop.save(os.path.join(result_path, img_crop_fn))
                img_crop = CreateCroppedImage(depth_img, seg.minx, seg.miny, seg.maxx, seg.maxy, fill_col=0, scl_fac=5, truncate=True)    # global crop
                img_crop = img_crop.resize((224,224), resample=PIL.Image.BICUBIC)
                img_crop_fn = depth_img_fn[0:len(depth_img_fn)-len('.png')]+'_seg_{}_g_depth'.format(seg_id)+'.pgm'
                img_crop.save(os.path.join(result_path, img_crop_fn))
                # Crop normal map
                img_crop = CreateCroppedImage(normal_img, seg.minx, seg.miny, seg.maxx, seg.maxy, fill_col=black_color)   # local crop
                img_crop = img_crop.resize((224,224), resample=PIL.Image.BICUBIC)
                img_crop_fn = normal_img_fn[0:len(normal_img_fn)-len('.jpg')]+'_seg_{}_l_normal'.format(seg_id)+'.png'
                img_crop.save(os.path.join(result_path, img_crop_fn))
                img_crop = CreateCroppedImage(normal_img, seg.minx, seg.miny, seg.maxx, seg.maxy, fill_col=black_color, scl_fac=5, truncate=True)    # global crop
                img_crop = img_crop.resize((224,224), resample=PIL.Image.BICUBIC)
                img_crop_fn = normal_img_fn[0:len(normal_img_fn)-len('.jpg')]+'_seg_{}_g_normal'.format(seg_id)+'.png'
                img_crop.save(os.path.join(result_path, img_crop_fn))
                # Crop segment mask image
                img_mask_crop = CreateCroppedImage(seg.mask, seg.minx, seg.miny, seg.maxx, seg.maxy, fill_col=False)   # local crop
                img_mask_crop = img_mask_crop.resize((224,224), resample=PIL.Image.BICUBIC)
                img_mask_crop_fn = col_img_fn[0:len(col_img_fn)-len('.jpg')]+'_seg_{}_l_mask'.format(seg_id)+'.png'
                img_mask_crop.save(os.path.join(result_path, img_mask_crop_fn))
                img_mask_crop = CreateCroppedImage(seg.mask, seg.minx, seg.miny, seg.maxx, seg.maxy, fill_col=False, scl_fac=5, truncate=True)    # global crop
                img_mask_crop = img_mask_crop.resize((224,224), resample=PIL.Image.BICUBIC)
                img_mask_crop_fn = col_img_fn[0:len(col_img_fn)-len('.jpg')]+'_seg_{}_g_mask'.format(seg_id)+'.png'
                img_mask_crop.save(os.path.join(result_path, img_mask_crop_fn))
                img_mask_cntx = CreateContextMask(img_mask_crop)
                img_mask_cntx_fn = col_img_fn[0:len(col_img_fn)-len('.jpg')]+'_seg_{}_g_mask_c'.format(seg_id)+'.png'
                img_mask_cntx.save(os.path.join(result_path, img_mask_cntx_fn))



thread_list = []
for tid in range(config.num_thread):
    sf = int(tid / num_thread * num_frame)
    ef = int((tid+1) / num_thread * num_frame)
    thread_list.append(myThread(tid, 't{}'.format(tid), tid, sf, ef))

# Start threads
for thread in thread_list:
    thread.start()

# Join threads
for thread in thread_list:
    thread.join()

print('All frames are done in {}'.format(strftime("%H:%M:%S",time.gmtime(time.time()-start))))