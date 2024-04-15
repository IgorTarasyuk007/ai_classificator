import tensorflow as tf
#import tensorflow.contrib.keras as keras
from tensorflow.keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report,confusion_matrix
from tensorflow import keras
from PIL import Image
from pathlib import Path
import scipy
import os
import numpy as np
import matplotlib.pyplot as plt
from torchvision.datasets import ImageFolder
import torchvision.transforms as T
# collect directory
data_dir = Path('tf_files/')
print(data_dir)
transformer = T.Compose([T.Resize((32, 32)), T.ToTensor()])
dataset = ImageFolder(data_dir, transform = transformer)

# display class names
print(dataset.classes)

imagepath_cups = r"tf_files/cups"
graypath_cups = r"processed_images/cups"
imagepath_bottles = r"tf_files/bottles"
graypath_bottles = r"processed_images/bottles"
imagepath_containers = r"tf_files/containers"
graypath_containers = r"processed_images/containers"

File_listing = os.listdir(imagepath_cups)
for file in File_listing:
    im = Image.open(imagepath_cups + '/' + file) 
    img = im.resize((32,32))
    gray = img.convert('L')
    gray.save(graypath_cups + '/' + file, "JPEG")
File_listing = os.listdir(imagepath_bottles)
for file in File_listing:
    im = Image.open(imagepath_bottles + '/' + file) 
    img = im.resize((32,32))
    gray = img.convert('L')
    gray.save(graypath_bottles + '/' + file, "JPEG")
File_listing = os.listdir(imagepath_containers)
for file in File_listing:
    im = Image.open(imagepath_containers + '/' + file) 
    img = im.resize((32,32))
    gray = img.convert('L')
    gray.save(graypath_containers + '/' + file, "JPEG")
'''
PATH_TEST = r"tf_files"
PATH_TRAIN = r"processed_images"

class_names = ['containers', 'cups', 'bottles']

train_dir = os.path.join(PATH_TRAIN)
test_dir = os.path.join(PATH_TEST)

imagepath_cups_dir = os.path.join(imagepath_cups)
imagepath_containers_dir = os.path.join(imagepath_containers)
imagepath_bottles_dir = os.path.join(imagepath_bottles)

IMG_HEIGHT = 32
IMG_WIDTH = 32

image_gen = ImageDataGenerator(rescale=1./255)

train_data_gen = image_gen.flow_from_directory(
    directory = train_dir, 
    shuffle=True, 
    target_size = (IMG_HEIGHT, IMG_WIDTH),
    class_mode='categorical')
test_data_gen = image_gen.flow_from_directory(
    directory = test_dir, 
    shuffle=True, 
    target_size = (IMG_HEIGHT, IMG_WIDTH),
    class_mode='categorical')
sample_data_gen = image_gen.flow_from_directory(
    directory = test_dir, 
    shuffle=True, 
    target_size = (180, 180),
    class_mode='categorical')

model = Sequential([
    Conv2D(filters=32, kernel_size=3, padding='same', activation='relu', input_shape=(IMG_HEIGHT,IMG_WIDTH, 3)),
    MaxPooling2D(pool_size=2),

    Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'),
    MaxPooling2D(pool_size=2),
    
    Conv2D(filters=32, kernel_size=3, padding='same', activation='relu'),
    MaxPooling2D(pool_size=2),
    
    Conv2D(filters=32, kernel_size=3, padding='same', activation='relu'),
    MaxPooling2D(pool_size=2),

    Flatten(),

    Dense(6, activation='softmax')
])
batch_size = 45
epochs = 50
model.compile(optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy'])

num_cups_train = len(os.listdir(imagepath_cups_dir))
num_containers_train = len(os.listdir(imagepath_containers_dir))
num_bottles_train = len(os.listdir(imagepath_bottles_dir))

num_cups_test = len(os.listdir(graypath_cups))
num_bottles_test = len(os.listdir(graypath_bottles))
num_containers_test = len(os.listdir(graypath_containers))

total_train = num_bottles_train + num_containers_train + num_cups_train
total_test = num_containers_test + num_bottles_test + num_cups_test
model.add(Dense(3, activation = "softmax"))
'''
'''
history = model.fit(
    train_data_gen,
    validation_data = train_data_gen,
    steps_per_epoch= total_train // batch_size,
    epochs = epochs,
    validation_steps= total_test // batch_size,
    callbacks = [tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                min_delta=0.01,
                patience=4)]
)
'''
'''
history = model.fit(
    train_data_gen,
    validation_data = train_data_gen,
    steps_per_epoch= total_train // batch_size,
    epochs = epochs,
    validation_steps= total_test // batch_size
)
test_loss, test_acc = model.evaluate(test_data_gen)
print('Test accuracy: {} Test Loss: {} '.format(test_acc*100, test_loss))
model.save("my_model"+str(int(test_acc*100))+".h5")
'''