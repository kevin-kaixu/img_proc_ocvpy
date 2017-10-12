import numpy as np
import seaborn as sns
from scipy import ndimage
import matplotlib.pyplot as plt
import PIL
import PIL.Image as Image
import PIL.ImageOps as ImgOps
import os

color_path = './data/'
depth_path = './data/'
normal_path = './data/Normal/'
seg_path = './data/segmentation_images/'
input_path = [color_path, depth_path, normal_path, seg_path]

result_path = './result/'
result_mask_path = './result/mask/'

class SegInfo:
    def __init__(self, minx, miny, maxx, maxy, width, height):
        self.minx = minx
        self.miny = miny
        self.maxx = maxx
        self.maxy = maxy
        self.numpnt = 0
        self.mask = Image.new('1', size=(width, height))

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
                pixel_map[x,y] = True
                seg_info_list[seg_id].numpnt += 1
                asm_map[x,y] = cpi[seg_id % num_color]
    
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
            # The region beyong the boundary is not truncated by filled by grey
            #   since we are using the desired crop size
            if cross_x == -1:
                shift[0] = img_xmin - crop_xmin
            if cross_y == -1:
                shift[1] = img_ymin - crop_ymin
        new_img = Image.new(img.mode, (crop_size, crop_size), fill_col)
        new_img.paste(img.crop((crop_xmin, crop_ymin, crop_xmax, crop_ymax)), tuple(shift))
        return new_img


filenames_list = GetFileNames(color_path, normal_path, seg_path)
os.makedirs(result_path, exist_ok=True)
os.makedirs(result_mask_path, exist_ok=True)

grey_color = (128, 128, 128)
black_color = (0, 0, 0)

#for i in range(len(filenames_list[0])):
for i in range(1):
    # Get segmentation information
    seg_info_list, all_seg_mask = GetSegmentsInfo(seg_path+filenames_list[3][i])
    # Open color image and generate segmentation masked color image
    col_img_fn = filenames_list[0][i]
    col_img = Image.open(color_path+col_img_fn)
    img_mask = Image.blend(col_img, all_seg_mask, alpha=0.7)
    img_mask_fn = col_img_fn[0:len(col_img_fn)-len('.jpg')]+'_mask.jpg'
    img_mask.save(result_mask_path+img_mask_fn)
    # Open depth image
    depth_img_fn = filenames_list[1][i]
    depth_img = Image.open(depth_path+depth_img_fn)
    # Open normal map
    normal_img_fn = filenames_list[2][i]
    normal_img = Image.open(normal_path+normal_img_fn)
    # For each segment, generate cropped color image, depth image, normal map and segment mask
    for seg, seg_id in zip(seg_info_list, range(len(seg_info_list))):
        if seg.numpnt < 300:
            print('  >>> Segment {} of image {} is being omitted due to too small size.'.format(seg_id, i))
            continue
        # Crop color image
        img_crop = CreateCroppedImage(col_img, seg.minx, seg.miny, seg.maxx, seg.maxy, fill_col=grey_color)   # local crop
        img_crop = img_crop.resize((224,224), resample=PIL.Image.BICUBIC)
        img_crop_fn = col_img_fn[0:len(col_img_fn)-len('.color.jpg')]+'_seg_{}_l_col'.format(seg_id)+'.png'
        img_crop.save(result_path+img_crop_fn)
        img_crop = CreateCroppedImage(col_img, seg.minx, seg.miny, seg.maxx, seg.maxy, fill_col=grey_color, scl_fac=5, truncate=True)    # global crop
        img_crop = img_crop.resize((224,224), resample=PIL.Image.BICUBIC)
        img_crop_fn = col_img_fn[0:len(col_img_fn)-len('.color.jpg')]+'_seg_{}_g_col'.format(seg_id)+'.png'
        img_crop.save(result_path+img_crop_fn)
        # Crop depth image
        img_crop = CreateCroppedImage(depth_img, seg.minx, seg.miny, seg.maxx, seg.maxy, fill_col=0)   # local crop
        img_crop = img_crop.resize((224,224), resample=PIL.Image.BICUBIC)
        img_crop_fn = depth_img_fn[0:len(depth_img_fn)-len('.depth.pgm')]+'_seg_{}_l_depth'.format(seg_id)+'.pgm'
        img_crop.save(result_path+img_crop_fn)
        img_crop = CreateCroppedImage(depth_img, seg.minx, seg.miny, seg.maxx, seg.maxy, fill_col=0, scl_fac=5, truncate=True)    # global crop
        img_crop = img_crop.resize((224,224), resample=PIL.Image.BICUBIC)
        img_crop_fn = depth_img_fn[0:len(depth_img_fn)-len('.depth.pgm')]+'_seg_{}_g_depth'.format(seg_id)+'.pgm'
        img_crop.save(result_path+img_crop_fn)
        # Crop normal map
        img_crop = CreateCroppedImage(normal_img, seg.minx, seg.miny, seg.maxx, seg.maxy, fill_col=black_color)   # local crop
        img_crop = img_crop.resize((224,224), resample=PIL.Image.BICUBIC)
        img_crop_fn = normal_img_fn[0:len(normal_img_fn)-len('.depth.jpg')]+'_seg_{}_l_normal'.format(seg_id)+'.png'
        img_crop.save(result_path+img_crop_fn)
        img_crop = CreateCroppedImage(normal_img, seg.minx, seg.miny, seg.maxx, seg.maxy, fill_col=black_color, scl_fac=5, truncate=True)    # global crop
        img_crop = img_crop.resize((224,224), resample=PIL.Image.BICUBIC)
        img_crop_fn = normal_img_fn[0:len(normal_img_fn)-len('.depth.jpg')]+'_seg_{}_g_normal'.format(seg_id)+'.png'
        img_crop.save(result_path+img_crop_fn)
        # Crop segment mask image
        img_mask_crop = CreateCroppedImage(seg.mask, seg.minx, seg.miny, seg.maxx, seg.maxy, fill_col=False)   # local crop
        img_mask_crop = img_mask_crop.resize((224,224), resample=PIL.Image.BICUBIC)
        img_mask_crop_fn = col_img_fn[0:len(col_img_fn)-len('.color.jpg')]+'_seg_{}_l_mask'.format(seg_id)+'.png'
        img_mask_crop.save(result_path+img_mask_crop_fn)
        img_mask_crop = CreateCroppedImage(seg.mask, seg.minx, seg.miny, seg.maxx, seg.maxy, fill_col=False, scl_fac=5, truncate=True)    # global crop
        img_mask_crop = img_mask_crop.resize((224,224), resample=PIL.Image.BICUBIC)
        img_mask_crop_fn = col_img_fn[0:len(col_img_fn)-len('.color.jpg')]+'_seg_{}_g_mask'.format(seg_id)+'.png'
        img_mask_crop.save(result_path+img_mask_crop_fn)
        img_mask_cntx = CreateContextMask(img_mask_crop)
        img_mask_cntx_fn = col_img_fn[0:len(col_img_fn)-len('.color.jpg')]+'_seg_{}_g_mask_c'.format(seg_id)+'.png'
        img_mask_cntx.save(result_path+img_mask_cntx_fn)


