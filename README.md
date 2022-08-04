# Computer Vision - Firearm Length Measurement
Computer vision method for measuring the overall length of a firearm in a picture

## Context


## How does it work ? 


## How to use it ?
### **1. Load the pre-trained model of Stacked Cross Refinement Network (*SCRN*)**
- research paper
- link

### **2. Put your image(s) in a specific folder**
- data/images_root
- another one

### **3. Launch the algorithm**
- If you wish to use the default folders to store your images ➡️ *./data/input_images* & *./data/segmented_images* : 
    
    `python3 main.py`
- Otherwise, you must specify these parameters :
    
    `python3 main.py --images_root ../yourImagesPath/ --save_root ../yourSavePath/`
