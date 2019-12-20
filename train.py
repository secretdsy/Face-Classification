import os
import warnings
import numpy as np 
import pandas as pd
import multiprocessing
from pathlib import Path

from keras import backend as K
warnings.filterwarnings(action='ignore')
K.image_data_format()

from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, UpSampling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model

from keras.applications.resnet_v2 import ResNet101V2, ResNet152V2
from keras.applications.nasnet import NASNetLarge, preprocess_input
from keras.callbacks import ModelCheckpoint, EarlyStopping

## functional
# from keras.layers import Input, Dense
# from keras.models import Model
# from keras.layers.merge import concatenate, add


data_dir = Path('./data')
# raw_dir = data_dir / 'raw/train'
# test_dir = data_dir / 'raw'
train_dir = data_dir / 'train'
val_dir = data_dir / 'val'
test_dir = data_dir
# aug_dir = data_dir / 'augmentation/train'
submit_dir = Path('./submit')
save_dir = Path('./savemodel')
label_dir = Path('./label')
train_path = label_dir / 'train_vision.csv'
test_path = label_dir / 'test_vision.csv'

params = {
    # Generator Parameter
    'random_state': 42,
    'horizontal_flip': True,
    'rotation_range': 20,
    'width_shift_range': 0.20,
    'height_shift_range': 0.20,
    'shear_range': 0.10,
    'zoom_range': 0.20,
    'brightness_range': (0.7, 1.3),

    # Model Parameter
    'img_size': (128, 128),
    'input_shape': (128, 128, 3),
    'batch_size': 8,
    'epochs': 100,
    'nb_workers': multiprocessing.cpu_count() // 2
}

train_datagen = ImageDataGenerator(
    rotation_range=params['rotation_range'],
    width_shift_range=params['width_shift_range'],
    height_shift_range=params['height_shift_range'],
    shear_range=params['shear_range'],
    zoom_range=params['zoom_range'],
    horizontal_flip=params['horizontal_flip'],
    brightness_range=params['brightness_range'],
    preprocessing_function=preprocess_input)
#     validation_split=0.8)

validation_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input)

train_generator = train_datagen.flow_from_directory(
    str(train_dir),
#     subset='training',
    target_size=params['img_size'],
    color_mode='rgb',
    class_mode='categorical',
    batch_size=params['batch_size'],
    seed=params['random_state'])

validation_generator = validation_datagen.flow_from_directory(
    str(val_dir),
#     subset='validation',
    target_size=params['img_size'],
    color_mode='rgb',
    class_mode='categorical',
    batch_size=params['batch_size'],
    shuffle=False)

def get_steps(num_samples, batch_size):
    if (num_samples % batch_size) > 0 :
        return (num_samples // batch_size) + 1
    else:
        return num_samples // batch_size

model = None
cnn_model = NASNetLarge(include_top=False, weights=None, input_shape=(256,256,3))
model = Sequential()
model.add(UpSampling2D(size=(2,2), input_shape=params['input_shape'], interpolation='bilinear'))
model.add(cnn_model)
model.add(GlobalAveragePooling2D())
model.add(Dense(6, activation='softmax', kernel_initializer='he_normal'))

model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.Adam(),
              metrics=['acc'])


filepath = str(save_dir / 'nasnet_upsam_aug_ep{epoch:03d}_acc-{acc:.4f}_vloss-{val_loss:.4f}_vacc-{val_acc:.4f}.h5')
# model = load_model(str(save_dir / 'res152_aug8+real8_val2_ep040_acc-0.9854_vloss-1.7443_vacc-0.9078.h5'))

checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True)
earlystop = EarlyStopping(monitor='val_acc', min_delta=0, patience=15, verbose=1, mode='auto')

callbacks = [checkpoint, earlystop]
# callbacks = [checkpoint]
model.save(filepath)

model.fit_generator(
    train_generator,
    steps_per_epoch = get_steps(train_generator.samples, params['batch_size']),
    validation_data = validation_generator, 
    validation_steps = get_steps(validation_generator.samples, params['batch_size']),
    callbacks = callbacks,
#     initial_epoch = 22,
    workers=params['nb_workers'],
    epochs = params['epochs'])