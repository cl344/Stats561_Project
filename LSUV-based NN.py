import os, shutil
import pandas as pd
import numpy as np
# os.makedirs('test_set')
source = "/Users/zhangjing/Work/ECE682 Prob ML/driver/imgs/train_set"
dest = "/Users/zhangjing/Work/ECE682 Prob ML/driver/imgs/test_set"
dirs = os.listdir(source)

for d in dirs:
    if (d != '.DS_Store'):
        src = source + '/' + d
        dst = dest + '/' + d
        os.mkdir(dst)
        files = os.listdir(src)
        for f in files:
            if np.random.rand(1) < 0.1:
                shutil.move(src + '/' + f, dst + '/' + f)

for d in dirs:
    if (d != '.DS_Store'):
        src = source + '/' + d
        dst = dest + '/' + d
        os.mkdir(dst)
        files = os.listdir(src)
        for f in files:
            if np.random.rand(1) < 0.02:
                shutil.move(src + '/' + f, dst + '/' + f)

# Convolutional Neural Network

# Installing Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Installing Tensorflow
# pip install tensorflow

# Installing Keras
# pip install --upgrade keras

# Part 1 - Building the CNN

# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, GlobalMaxPooling2D
from keras.layers import Flatten, Dense, Dropout, Activation 

# Model
model = Sequential()

model.add(Conv2D(32, (3, 3), padding = 'same', input_shape = (224, 224, 3), activation = 'relu'))
model.add(Conv2D(32, (3, 3), padding = 'same', input_shape = (224, 224, 3), activation = 'relu'))
model.add(Conv2D(32, (3, 3), padding = 'same', input_shape = (224, 224, 3), activation = 'relu'))
model.add(Conv2D(32, (3, 3), padding = 'same', input_shape = (224, 224, 3), activation = 'relu'))
model.add(Conv2D(32, (3, 3), padding = 'same', input_shape = (224, 224, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(80, (3, 3), padding='same', input_shape = (224, 224, 3), activation = 'relu'))
model.add(Conv2D(80, (3, 3), padding='same', input_shape = (224, 224, 3), activation = 'relu'))
model.add(Conv2D(80, (3, 3), padding='same', input_shape = (224, 224, 3), activation = 'relu'))
model.add(Conv2D(80, (3, 3), padding='same', input_shape = (224, 224, 3), activation = 'relu'))
model.add(Conv2D(80, (3, 3), padding='same', input_shape = (224, 224, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, (3, 3), padding='same', input_shape = (224, 224, 3), activation = 'relu'))
model.add(Conv2D(128, (3, 3), padding='same', input_shape = (224, 224, 3), activation = 'relu'))
model.add(Conv2D(128, (3, 3), padding='same', input_shape = (224, 224, 3), activation = 'relu'))
model.add(Conv2D(128, (3, 3), padding='same', input_shape = (224, 224, 3), activation = 'relu'))
model.add(Conv2D(128, (3, 3), padding='same', input_shape = (224, 224, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(500))
model.add(Activation('relu'))
model.add(Dropout(0.4))
model.add(Dense(10))
model.add(Activation('softmax'))

opt = keras.optimizers.Adam(lr = 0.0001, decay=1e-6)
model.compile(optimizer = opt, loss = 'categorical_crossentropy', metrics = ['accuracy'])


from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

train_generator = train_datagen.flow_from_directory('./train_set/',
                                                 target_size = (224, 224),
                                                 color_mode = "rgb",
                                                 batch_size = 32,
                                                 class_mode = 'categorical',
                                                 shuffle = True,
                                                 seed = 2019);

valid_generator = test_datagen.flow_from_directory('./valid_set/',
                                            target_size = (224, 224),
                                            color_mode = "rgb",
                                            batch_size = 32,
                                            class_mode = 'categorical',
                                            shuffle = True,
                                            seed = 2019);

STEP_SIZE_TRAIN = train_generator.n//train_generator.batch_size;
STEP_SIZE_VALID = valid_generator.n//valid_generator.batch_size;

classifier.fit_generator(train_generator,
                         steps_per_epoch = STEP_SIZE_TRAIN,
                         epochs = 20,
                         validation_data = valid_generator,
                         validation_steps = STEP_SIZE_VALID);
                         
                         
