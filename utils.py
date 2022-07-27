import os

from datetime import datetime
import numpy as np

import cv2
import imageio
import skimage

import torch
import torch.nn.functional as F
from torch.autograd import Variable

from SCRN.model.ResNet_models import SCRN

SCRN_SIZE = 352



def prepare_image(image_path: str) -> torch.Tensor:
    """ Convert a PIL Image to model-compatible input
    Args:
        image_path (str): path of the input image
    Returns:
        t_image (torch.Tensor): converted image, shape = (3, SCRN_SIZE, SCRN_SIZE), normalized on ImageNet
        original_size (list): list containing the original size (width, height) of the image (int, int)
    """        
    img_transform = transforms.Compose([
            transforms.Resize((SCRN_SIZE, SCRN_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]) # 3 values of mean and std : for each channel of RGB
    
    with open(path, 'rb') as f:
        im = Image.open(f)
        image = im.convert('RGB')
    
    original_size = image.size
    t_image = img_transform(image).unsqueeze(0)
    
    return t_image, original_size

    
    
def load_SCRN():
    """ Load SCRN model from its .pth file
    Returns:
        model (Model): loaded model ready for prediction
    """    
    device = torch.device('cpu')
    model = SCRN()
    model.load_state_dict(torch.load('./SCRN/model/model.pth', map_location=device))
    model.to(device)
    model.eval()
    
    return model



def apply_SCRN(model, image: torch.Tensor, image_name: str, original_size: list) -> str:
    """ Apply SCRN on the image AND save the resulted image with its corresponding length
    Args:
        model (Model): loaded model
        image (torch.Tensor): preprocessed image
        image_name (str): name of the image in the folder
        original_size (list): list containing the original size (width and height) of the image
    Returns:
        new_path (str): path of the segmentated image made by SCRN
    """   
    image = Variable(image).cpu()

    res, _ = model(image) # apply SCRN on the preprocessed image
    res = F.upsample(res, size=(original_size[1], original_size[0]), mode='bilinear', align_corners=True) # resizing the result
    res = res.sigmoid().data.cpu().numpy().squeeze() # dtype : float32
    res = skimage.img_as_ubyte(res) # dtype : uint8 (this type is needed by imagio.imwrite())

    new_path = save_path + image_name + '.png'
    imageio.imwrite(new_path, res)
    
    return new_path
    

    
def measure_length(image_path: str, new_path: str, original_size: list, preproc_image: torch.Tensor) -> float:
    """ Measure the length of the gun in the image
    Args:
        image_path (str): path of the original image
        new_path (str): path of the segmented image
        original_size (list): width and height of the original image (before preprocessing)
        preproc_image (torch.Tensor): preprocessed tensor (resized, normalized)
    Returns:
        length (float): length of the gun in centimeters
    """    
    orig = cv2.imread(image_path) # original image
    orig = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)

    im_preproc = preproc_image[0].permute(1,2,0).numpy().astype(np.uint8) # preprocessed image from SCRN
    im_preproc = cv2.resize(im_preproc, (original_size[0], original_size[1]))

    im = cv2.imread(new_path) # segmented image from SCRN
    # im = cv2.erode(im, kernel=np.ones((5,5), np.uint8), iterations=2)

    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1] 
    thresh = cv2.bitwise_not(thresh)

    canny = cv2.Canny(thresh, 50, 100, apertureSize=3)
    canny = cv2.dilate(canny, kernel=np.ones((3,3), np.uint8), iterations=1)
    lines = cv2.HoughLinesP(canny, rho=1, theta=1*np.pi/180, threshold=200, minLineLength=10, maxLineGap=100) 
    orig_lines = orig.copy()
    if lines is not None:
        for line in lines: # Draw lines on the image
            print(line, type(line))
            x1, y1, x2, y2 = line[0]
            cv2.line(orig_lines, (x1, y1), (x2, y2), (255, 0, 0), 3)



    cnts, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # take a binary image as parameter
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    max_cnt = cnts[0]


    orig_contour = orig.copy()
    cv2.drawContours(orig_contour, cnts, 0, (0,255,0), 5)


    orig_rect = orig.copy()
    d = {'x':[], 'y':[], 'w':[], 'h':[], 'wh':[]}
    for c in cnts: 
        x, y, w, h = cv2.boundingRect(c) # returns parameters of the bounding box of a contour
        d['x'].append(x)
        d['y'].append(y)
        d['w'].append(w)
        d['h'].append(h)
        d['wh'].append(w*h)
        


    for k in range(1): # keep only the k largest areas
        i = d['wh'].index(max(d['wh'])) # find the largest rectangle
        d['wh'][i] = 0 # we then set it to 0 in order to choose another one during next loop
        cv2.rectangle(orig_rect, (d['x'][i], d['y'][i]), (d['x'][i]+d['w'][i], d['y'][i]+d['h'][i]), (0, 0, 255), 5)


    orig_rot = orig.copy()
    rect = cv2.minAreaRect(max_cnt) # rect = [(x,y), (width,height), angle]
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    cv2.drawContours(orig_rot, [box], 0, (0,0,255), 5)



    left = tuple(max_cnt[max_cnt[:, :, 0].argmin()][0])
    right = tuple(max_cnt[max_cnt[:, :, 0].argmax()][0])
    bottom = tuple(max_cnt[max_cnt[:, :, 1].argmax()][0])
    top = tuple(max_cnt[max_cnt[:, :, 1].argmin()][0])
    ext = [left, right, bottom, top]

    for e in ext:
        cv2.circle(orig_contour, e, 8, (255,0,0), -1)
        
    length = None
    return length



def BasegunV3(images_root: str, save_root: str) -> dict:
    """ Load and apply SCRN on each image after preprocessing AND compute the length of the gun in every image
    Args:
        images_root (str): path of the folder containing all images
        save_root (str): path of the folder that will contain the segmented images (segmentation by SCRN)
    Returns:
        lengths (dict): contains the lengths (value) from each image (key)
    """

    if not os.path.exists(save_root):
        os.makedirs(save_root)

    model = load_model() # SCRN
        
    lengths = {}
    
    with torch.no_grad():
        for image_name in os.listdir(images_root):
            image_path = images_root + '/' + image_name
            
            preproc_image, original_size = prepare_image(image_path) # preprocessed tensor (resized, normalized) & original size of the image
            
            new_path = apply_SCRN(model, preproc_image, image_name, original_size) # path of the segmented image (segmentation made by SCRN)

            lengths[image_path] = measure_length(image_path, new_path, original_size, preproc_image)
            
    return lengths