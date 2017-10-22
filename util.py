from argparse import ArgumentParser

def get_args():
    parser = ArgumentParser(description='training_img_gen')
    parser.add_argument('--tol_size', type=int, default=300)
    parser.add_argument('--print_every', type=int, default=1)
    parser.add_argument('--no_print', action='store_true', default=False)
    parser.add_argument('--blend_alpha', type=float, default=0.7)
    parser.add_argument('--num_col', type=int, default=32)
    parser.add_argument('--num_thread', type=int, default=8)
    parser.add_argument('--no_config_file', action='store_true', default=False)
    parser.add_argument('--config_file', type=str, default='config.txt')
    parser.add_argument('--color_path', type=str, default='./data/color')
    parser.add_argument('--depth_path', type=str, default='./data/depth')
    parser.add_argument('--normal_path', type=str, default='./data/normal')
    parser.add_argument('--seg_path', type=str, default='./data/segmentation_images')
    parser.add_argument('--camera_path', type=str, default='./data')
    parser.add_argument('--result_path', type=str, default='./data/result')
    parser.add_argument('--result_mask_path', type=str, default='./data/result/mask')
    args = parser.parse_args()
    return args



def get_config_from_file(config_file_name, config):
    new_config = config
    with open(config_file_name) as file:
        for line in file:
            attr, value = line.split()
            if hasattr(new_config, attr):
                setattr(new_config, attr, eval(value))
    file.close()
    return new_config