# Computer Vision - Firearm Length Measurement
Computer vision method for measuring the overall length of a firearm in a picture

## How does it work ? 


## How to use it ?
### **1. Load the pre-trained model of Stacked Cross Refinement Network (*SCRN*)**
- research paper
- link

### **2. Put your image(s) in a specific folder**
- data/images_root
- another one

### **3. Launch the algorithm**
- If you want to use the folders by default (./data/input_images & ./data/segmented_images) : 
    python3 main.py
- Otherwise, you must specify these two parameters :
    python3 main.py --images_root .../yourImagesPath/ --save_root .../yourSavePath/
