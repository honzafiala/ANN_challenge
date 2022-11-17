import os
import random
import shutil
import sys
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import visualkeras
from PIL import Image
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator

tfk = tf.keras
tfkl = tf.keras.layers

if len(sys.argv[1]) < 2:
    print("Expecting model file name as argument")
    exit(0)


# Random seed for reproducibility
seed = 42

random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
tf.random.set_seed(seed)
tf.compat.v1.set_random_seed(seed)


# Dataset folders 
dataset_dir = 'dataset'
training_dir = 'training'
validation_dir = 'validation'


labels = ['Species1',   # 0
          'Species2',   # 1
          'Species3',   # 2
          'Species4',   # 3
          'Species5',   # 4
          'Species6',   # 5
          'Species7',   # 6
          'Species8',   # 7
]

# count the occurrence for each class
#find the max occurrence of sample in a class
occurrenciences = {}
max = ['Label', 0]
directory = os.path.join(dataset_dir, training_dir)
for filename in os.listdir(directory):
    f = os.path.join(directory, filename)
    # checking if it is a file
    if os.path.isfile(f):
        print("File: " + f)
    else:
        count = 0
        for samples in os.listdir(f):
            s = os.path.join(f, samples)
            if os.path.isfile(s):
                count+=1
        occurrenciences[f.split('/')[2]] = count
        if count > max[1]:
          max[0] = f.split('/')[2]
          max[1] = count
print("\n\nClasses occurences: ", end='') 
print(occurrenciences)

print("Max occurences: ", end='')
print(max)

# compute 30% for each class to create the validation set 
percentage = 0.2
size_val_no_aug = {}
for i in occurrenciences:
    size_val_no_aug[i] = round(int(occurrenciences[i])*percentage)
print("Size of the sample for validation set (" + str(percentage*100) + "%) without augmenting: ", end='')
print(size_val_no_aug)
print("Size of the sample for validation set (" + str(percentage*100) + "%) with augmenting: ", end='')
print(str(round(max[1]*percentage)), end='\n\n')


#create the validation set directory if it doesn't exit for non augmented dataset
os.makedirs(dataset_dir+'/'+validation_dir, exist_ok=True)
for i in labels:
    os.makedirs(dataset_dir+'/'+validation_dir+'/'+i, exist_ok=True)

#move samples to the validation set
if 'VAL' in sys.argv:
    for i in occurrenciences:
        file_dir = dataset_dir+'/'+training_dir+'/'+i+'/'
        new_dir = dataset_dir+'/'+validation_dir+'/'+i+'/'
        max = size_val_no_aug[i]
        count = 0
        for file in os.listdir(file_dir):
            if os.path.isfile(file_dir + file) and count<max:
                shutil.move(file_dir + file, new_dir + file)
            else:
                break
            count += 1

import logging
#Suppress warnings
import warnings

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=Warning)
tf.get_logger().setLevel('INFO')
tf.autograph.set_verbosity(0)

tf.get_logger().setLevel(logging.ERROR)
tf.get_logger().setLevel('ERROR')
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# Images are divided into folders, one for each class. 
# If the images are organized in such a way, we can exploit the 
# ImageDataGenerator to read them from disk.

# Create an instance of ImageDataGenerator for training, validation, and test sets
aug_train_data_gen = ImageDataGenerator(
                                    vertical_flip=True, 
                                    fill_mode='reflect',
                                    )
train_data_gen = ImageDataGenerator()
valid_data_gen = ImageDataGenerator()


# shuffle operate at each epoc
# Obtain a data generator with the 'ImageDataGenerator.flow_from_directory' method
'''
aug_train_gen = aug_train_data_gen.flow_from_directory(directory=dataset_dir+'/'+training_dir,
                                               target_size=(96, 96),
                                               color_mode='rgb',
                                               classes=None, # can be set to labels
                                               class_mode='categorical',
                                               batch_size=8,
                                               shuffle = True,
                                               seed = seed)
'''
train_gen = train_data_gen.flow_from_directory(directory=dataset_dir+'/'+training_dir,
                                               target_size=(96, 96),
                                               color_mode='rgb',
                                               classes=None, # can be set to labels
                                               class_mode='categorical',
                                               batch_size=8,
                                               shuffle = True,
                                               seed =seed)
valid_gen = valid_data_gen.flow_from_directory(directory=dataset_dir+'/'+validation_dir,
                                               target_size=(96, 96),
                                               color_mode='rgb',
                                               classes=None, # can be set to labels
                                               class_mode='categorical',
                                               batch_size=8,
                                               shuffle=False,
                                               seed=seed)

def get_next_batch(generator):
  batch = next(generator)

  image = batch[0]
  target = batch[1]

  print("(Input) image shape:", image.shape)
  print("Target shape:",target.shape)

  # Visualize only the first sample
  image = image[0]
  target = target[0]
  target_idx = np.argmax(target)
  print()
  print("Categorical label:", target)
  print("Label:", target_idx)
  print("Class name:", labels[target_idx])
  fig = plt.figure(figsize=(6, 4))
  plt.imshow(np.uint8(image))
  

  return batch

## Model metadata
input_shape = (96, 96, 3)
epochs = 200

def build_model_fine_tuning(input_shape):
    ft_model = tfk.models.load_model(sys.argv[1])

    ft_model.get_layer('vgg16').trainable = True
    for i, layer in enumerate(ft_model.get_layer('vgg16').layers):
        print(i, layer.name, layer.trainable)

    for i, layer in enumerate(ft_model.get_layer('vgg16').layers[:14]):
        layer.trainable=False
    for i, layer in enumerate(ft_model.get_layer('vgg16').layers):
        print(i, layer.name, layer.trainable)
    ft_model.summary()

    ft_model.compile(loss=tfk.losses.CategoricalCrossentropy(), optimizer=tfk.optimizers.Adam(1e-4), metrics='accuracy')

    return ft_model


# Utility function to create folders and callbacks for training
from datetime import datetime

# Build model (for NO augmentation training)
model = build_model_fine_tuning(input_shape)
model.summary()

# Create folders and callbacks and fit
noaug_callbacks = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=20, restore_best_weights=True)

# Train the model
history = model.fit(
    x = train_gen,
    epochs = epochs,
    validation_data = valid_gen,
    callbacks = noaug_callbacks,
).history

visualkeras.layered_view(model, legend=True, spacing=20, scale_xy=10)

# We have to create the validation set (train/validation ratio: 70/30) 
# After we can augment training set to balance it [!Important: Augment only training set, no validation set => we have to split before the validation set!]  {max samples for class: 537 (I think)}


# Save best epoch model
model.save("model.h5")