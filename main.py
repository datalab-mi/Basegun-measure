import sys
from utils import BasegunV3



def main(images_root='./data/input_images', save_root='./data/segmented_images') -> None:
    """ Return the images with the corresponding gun length
    Args:
        images_root (str): path of the folder containing all images
        save_root (str): path of the folder that will contain the segmented images (segmentation by SCRN)
    """
    
    lengths = BasegunV3(images_root, save_root)
        
        
        
if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
