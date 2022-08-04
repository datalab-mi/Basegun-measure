# Computer Vision - Firearm Length Measurement
Computer vision method for measuring the overall length of a firearm in a picture

## Context
▶️ This research work is part of a project called [Basegun], which is carried by Datalab : a team of Data scientists & Web developers who are working for the French Ministry of the Interior.

This web application was designed to support police officers during their daily work : its goal is to recognize firearms and provide as much information as possible about them, from a single picture. It is mainly based on Deep Learning & Computer Vision techniques.

Here is the official [repository] of Basegun.

▶️ I had the opportunity to do a 4-months internship at Datalab : my mission was to help the team improve the project.

One of the actions I have set up is this algorithm that measures a firearm length from a picture. 

[Basegun]: https://eig.etalab.gouv.fr/defis/basegun/
[repository]: https://github.com/datalab-mi/Basegun


## How does it work ? 
### **1. Salient Object Detection (SOD)**
I am using a SOD model called Stacked Cross Refinement Network (SCRN). It is a neural network based method that is working very well at detecting and segmenting the most salient object in a picture.

### **2. Computer Vision techniques**
Then, I implemented different computer vision techniques in order to measure the overall length of the gun.
The main idea is to find its length in pixel and then compare it with the pixel length of a well-known reference object (such as a credit card whose length in millimeters is known and constant for everyone).


## How to use it ?
### **1. Load the pre-trained model of Stacked Cross Refinement Network (*SCRN*)**
- Here is the [link] to their GitHub repository 
    ➡️ it contains the paper, the results of their experiments and a link to the [pre-trained] model
- Download *model.pth* and put it in *./SCRN/* 

[link]: https://github.com/wuzhe71/SCRN
[pre-trained]: https://drive.google.com/file/d/1PkGX9R-uTYpWBKX0lZRkE2qvvpz1-IiG/view


### **2. Put your image(s) in a specific folder**
This folder will contain images of firearms. There should be only 1 firearm for each image.
- By default, the folder is *./data/input_images*
- It can be another one : this new path will have to be given in parameters at the launch

### **3. Launch the algorithm**
- If you wish to use the default folders to store your images ➡️ *./data/input_images* & *./data/segmented_images* : 
    
    `python3 main.py`
- Otherwise, you should specify these parameters :
    
    `python3 main.py --images_root ../yourImagesPath/ --save_root ../yourSavePath/`
