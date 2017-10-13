import os
from argparse import ArgumentParser

def get_args():
    parser = ArgumentParser(description='training_img_gen')
    parser.add_argument('--tol_size', type=int, default=300)
    parser.add_argument('--print_every', type=int, default=1)
    parser.add_argument('--print', action='store_true', default=False)
    parser.add_argument('--blend_alpha', type=float, default=0.7)
    parser.add_argument('--col_num', type=int, default=32)
    parser.add_argument('--color_path', type=str, default='./data/')
    parser.add_argument('--depth_path', type=str, default='./data/')
    parser.add_argument('--normal_path', type=str, default='./data/Normal/')
    parser.add_argument('--seg_path', type=str, default='./data/segmentation_images/')
    parser.add_argument('--result_path', type=str, default='./result/')
    parser.add_argument('--result_mask_path', type=str, default='./result/mask/')
    args = parser.parse_args()
    return args