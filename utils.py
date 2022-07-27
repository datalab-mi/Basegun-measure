import os

from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np

import cv2
import imageio
import skimage

import torch
import torch.nn.functional as F
from torch.autograd import Variable

from SCRN.model.ResNet_models import SCRN
from SCRN.utils.data import test_dataset, SCRN_SIZE



def apply_SCRN(image_root):
    """ Load SCRN model with parameters and then apply it on the images of the folder 
    Args:
        image_root (str): path of the folder containing images
    Returns:
        infos (dict): contains pre-processed image, path of the segmented image and the original size
    """
    
    device = torch.device('cpu')
    model = SCRN()
    model.load_state_dict(torch.load('./SCRN/model/model.pth', map_location=device))
    model.to(device)
    model.eval()

    save_path = './SCRN/saliency_maps/save_test/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    test_loader = test_dataset(image_root, image_size=SCRN_SIZE, n_images=len(os.listdir(image_root)))

    infos = {}

    with torch.no_grad():
        for i in range(test_loader.size):
            image, name, prev_size = test_loader.load_data() # returns the i-th preprocessed image (resized, tensor, normalized), its path name, and its original size

            image = Variable(image).cpu()

            res, edge = model(image) # apply SCRN on the preprocessed image

            res = F.upsample(res, size=(prev_size[1],prev_size[0]), mode='bilinear', align_corners=True) # resizing the result

            res = res.sigmoid().data.cpu().numpy().squeeze() # dtype : float32
            res = skimage.img_as_ubyte(res) # dtype : uint8 (this type is needed by imagio.imwrite())

            new_path = save_path + name + '.png'
            imageio.imwrite(new_path, res)

            infos[i] = {"path": new_path, "size": prev_size, "preproc_image": image} # ///////!\\\\\\\\ change key to new_path
            # we store pre-processed image, path and original size in order to use them later
            
            
            # ?? resize image before returning
            # make all processing at the same time for lower complexity:
            
            
            # measure_length(...) 
    return infos



def measure_length(infos_image):
    
    orig = cv2.imread(os.path.join("/workspace/data/val/autre_epaule", infos_image["path"].replace('SCRN/saliency_maps/test//','').replace('png','jpg'))) # original image 
    orig = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)

    im_preproc = infos_image["preproc_image"][0].permute(1,2,0).numpy().astype(np.uint8) # preprocessed image from SCRN
    im_preproc = cv2.resize(im_preproc, (infos_image["size"][0], infos_image["size"][1]))

    im = cv2.imread(infos_image["path"]) # segmented image from SCRN
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