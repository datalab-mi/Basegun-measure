import argparse
import sys
from utils import BasegunV3



def main(images_root, save_root) -> None:
    """ Return the images with the corresponding gun length
    Args:
        images_root (str): path of the folder containing all images
        save_root (str): path of the folder that will contain the segmented images (segmentation by SCRN)
    """
    lengths = BasegunV3(images_root, save_root)
    
        
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--images_root', type=str, default='./data/input_images/', help='Path of the folder containing all images')
    parser.add_argument('--save_root', type=str, default='./data/segmented_images/', help='Path of the folder that will store the segmented images')
    opt = parser.parse_args()
    
    main(opt.images_root, opt.save_root)
