#!/usr/bin/env python
# coding: utf-8

# In[15]:
# All based on in-bed pose estimation data. https://github.com/ostadabbas/Seeing-Under-the-Cover

import cv2 as cv
import matplotlib.pyplot as plt
import os
import tensorflow as tf
import tensorflow as tf
import numpy as np

from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt




net = cv.dnn.readNetFromTensorflow("/Users/maddieenseleit/Research/graph_opt.pb") ## weights

inWidth = 368
inHeight = 368
thr = 0.2

BODY_PARTS = { "Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
                   "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
                   "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
                   "LEye": 15, "REar": 16, "LEar": 17, "Background": 18 }

POSE_PAIRS = [ ["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
                   ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
                   ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["Neck", "LHip"],
                   ["LHip", "LKnee"], ["LKnee", "LAnkle"], ["Neck", "Nose"], ["Nose", "REye"],
                   ["REye", "REar"], ["Nose", "LEye"], ["LEye", "LEar"] ]


# In[16]:


base_path = "/Users/maddieenseleit/Research/"
input_path = base_path + "Input/Train/uncover/" #CHOOSE UNCOVER, COVER1, COVER2
array = os.listdir(input_path)


real_pngs = []


def find_real_files():
    for img in array:
        temp = img.split(".")
        if temp[1] == 'png':
            real_pngs.append(input_path + img)

find_real_files()   



# In[17]:


for index, path in enumerate(real_pngs):
    print(index, path)
    image = cv.imread(path)
    plt.imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB)) 
    plt.show()
       


# In[ ]:


def pose_estimation(frame):
    frameWidth = frame.shape[1]
    frameHeight = frame.shape[0]
    
    net.setInput(cv.dnn.blobFromImage(frame, 1.0, (inWidth, inHeight), (127.5, 127.5, 127.5), swapRB= False, crop=False))
    out = net.forward()
    out = out[:, :19, :, :]
    
    assert(len(BODY_PARTS) == out.shape[1])
    
    points = []
    
    for i in range(len(BODY_PARTS)):
        # Slice heatmap of corresponding body's part.
        heatMap = out[0, i, :, :]

        # Originally, we try to find all the local maximums. To simplify a sample
        # we just find a global one. However only a single pose at the same time
        # could be detected this way.
        _, conf, _, point = cv.minMaxLoc(heatMap)
        x = (frameWidth * point[0]) / out.shape[3]
        y = (frameHeight * point[1]) / out.shape[2]
        

        # Add a point if it's confidence is higher than threshold.
        points.append((int(x), int(y)) if conf > thr else None)

    for pair in POSE_PAIRS:
        partFrom = pair[0]
        partTo = pair[1]
        assert(partFrom in BODY_PARTS)
        assert(partTo in BODY_PARTS)

        idFrom = BODY_PARTS[partFrom]
        idTo = BODY_PARTS[partTo]

        if points[idFrom] and points[idTo]:
            cv.line(frame, points[idFrom], points[idTo], (0, 255, 0), 3)
            cv.ellipse(frame, points[idFrom], (3, 3), 0, 0, 360, (0, 0, 255), cv.FILLED)
            cv.ellipse(frame, points[idTo], (3, 3), 0, 0, 360, (0, 0, 255), cv.FILLED)
        print(y)
        print(x)
        
    t, _ = net.getPerfProfile()
    freq = cv.getTickFrequency() / 1000
    cv.putText(frame, '%.2fms' % (t / freq), (10, 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

    return frame


# In[ ]:


for index, path in enumerate(real_pngs):
    print(index, path)
    pose = cv.imread(path)
    plt.imshow(cv.cvtColor(pose, cv.COLOR_BGR2RGB)) 
    estimated_image = pose_estimation(pose) 
    plt.imshow(cv.cvtColor(estimated_image, cv.COLOR_BGR2RGB))
    plt.imsave(f'{base_path}Output/{index}.png', cv.cvtColor(estimated_image, cv.COLOR_BGR2RGB))
    plt.show()


# In[ ]:




