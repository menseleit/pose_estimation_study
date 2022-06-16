#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# In[1]:


import os
import cv2
train_c1 = os.path.join (r"C:/Users/maddieenseleit/Research/Input/Train/cover1")
train_c2 = os.path.join (r"C:/Users/maddieenseleit/Research/Input/Train/cover2")
train_under = os.path.join (r"C:/Users/maddieenseleit/Research/Input/Train/uncover")


# In[1]:


import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.utils import shuffle


train_datagen = ImageDataGenerator(rescale=1/255)
validation_datagen = ImageDataGenerator(rescale=1/255)
train_generator = train_datagen.flow_from_directory(
   "/Users/maddieenseleit/Research/Input/Train",
    color_mode = "grayscale",
    target_size=(576, 1024),
    batch_size=128,
    class_mode='categorical',
    
)
validation_generator = validation_datagen.flow_from_directory(
    "/Users/maddieenseleit/Research/Input/Test",
    color_mode = "grayscale",
    target_size=(576, 1024),
    batch_size=32,
    class_mode ='categorical',
 
    
    
)

#Keep in?
#shuffle(train_generator)
#shuffle(validation_generator)

# In[3]:


img_rows, img_cols = 576, 1024

input_shape=(img_cols,img_rows,1)

from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from keras.layers import BatchNormalization


# In[4]:

  
model = tf.keras.models.Sequential([
    
    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape= input_shape),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.BatchNormalization(),
    
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.BatchNormalization(),
     
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.BatchNormalization(),
    
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.BatchNormalization(),
    
    # Flatten the results
    tf.keras.layers.Flatten(),
    
    tf.keras.layers.Dense(32, activation='softmax'),
    
    tf.keras.layers.Dense(3, activation='softmax')
])


# In[5]:


model.summary()


# In[6]:
from tensorflow import keras
from tensorflow.keras import layers
opt = keras.optimizers.Adam(learning_rate = 0.0001)
model.compile(loss="categorical_crossentropy",
             optimizer=opt,
             metrics=['accuracy'])


# In[ ]:


history = model.fit(
    train_generator,  
    steps_per_epoch=2,
    epochs=8,
    verbose=1,
    validation_data = validation_generator,
    validation_steps = 2
)


# In[ ]:


import matplotlib.pyplot as plt
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracies')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['train', 'test'], loc='best')
plt.show()



# In[ ]:


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['train', 'test'], loc='best')
plt.show()

