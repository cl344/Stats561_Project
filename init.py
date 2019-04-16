
import numpy as np

import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, GlobalMaxPooling2D
from keras.layers import Flatten, Dense, Dropout, Activation 
from keras.preprocessing.image import ImageDataGenerator
from keras.models import model_from_json
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import time

def modelWithInit(init):
    model = Sequential()
    
    model.add(Conv2D(32, (3, 3), padding = 'same', kernel_initializer = init, input_shape = (224, 224, 3), activation = 'relu'))
    model.add(Conv2D(32, (3, 3), padding = 'same', kernel_initializer = init, activation = 'relu'))
    model.add(Conv2D(32, (3, 3), padding = 'same', kernel_initializer = init, activation = 'relu'))
    model.add(Conv2D(32, (3, 3), padding = 'same', kernel_initializer = init, activation = 'relu'))
    model.add(Conv2D(32, (3, 3), padding = 'same', kernel_initializer = init, activation = 'relu'))
    model.add(MaxPooling2D(pool_size = (2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(80, (3, 3), padding='same', kernel_initializer = init, activation = 'relu'))
    model.add(Conv2D(80, (3, 3), padding='same', kernel_initializer = init, activation = 'relu'))
    model.add(Conv2D(80, (3, 3), padding='same', kernel_initializer = init, activation = 'relu')) 
    model.add(Conv2D(80, (3, 3), padding='same', kernel_initializer = init, activation = 'relu'))
    model.add(Conv2D(80, (3, 3), padding='same', kernel_initializer = init, activation = 'relu'))
    model.add(MaxPooling2D(pool_size = (2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, (3, 3), padding='same', kernel_initializer = init, activation = 'relu'))
    model.add(Conv2D(128, (3, 3), padding='same', kernel_initializer = init, activation = 'relu'))
    model.add(Conv2D(128, (3, 3), padding='same', kernel_initializer = init, activation = 'relu'))
    model.add(Conv2D(128, (3, 3), padding='same', kernel_initializer = init, activation = 'relu'))
    model.add(Conv2D(128, (3, 3), padding='same', kernel_initializer = init, activation = 'relu'))
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
    
    return model

class TimeHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)

def training(model):
    train_datagen = ImageDataGenerator(rescale = 1./255, shear_range = 0.2, zoom_range = 0.2, horizontal_flip = True)
    test_datagen = ImageDataGenerator(rescale = 1./255)
    
    train_generator = train_datagen.flow_from_directory('train_small/', target_size = (224, 224), color_mode = "rgb", batch_size = 32,
                                                 class_mode = 'categorical', shuffle = True, seed = 2019);
                                                        
    valid_generator = test_datagen.flow_from_directory('valid_small/', target_size = (224, 224), color_mode = "rgb", batch_size = 32, 
                                                       class_mode = 'categorical', shuffle = True, seed = 2019);
                                                       
    STEP_SIZE_TRAIN = train_generator.n//train_generator.batch_size;
    STEP_SIZE_VALID = valid_generator.n//valid_generator.batch_size;
    
    time_callback = TimeHistory()
    model.fit_generator(train_generator, steps_per_epoch = STEP_SIZE_TRAIN, epochs = 25, 
                        validation_data = valid_generator, validation_steps = STEP_SIZE_VALID, 
                        callbacks=[time_callback]);
    times = time_callback.times
    return (model, times)

# Four different initialization methods
init0 = keras.initializers.glorot_normal(seed = 2019)
init1 = keras.initializers.glorot_uniform(seed = 2019)
init2 = keras.initializers.he_normal(seed = 2019)
init3 = keras.initializers.he_uniform(seed = 2019)
inits = [init0, init1, init2, init3]

t = 0
for init in inits:
    t = t + 1
    model = modelWithInit(init)
    model, times = training(model)


    history=model.history
    print("initialization method {}".format(t))

    print(history['acc'])
    print(history['val_acc'])

    print(history['loss'])
    print(history['val_loss'])

    print("initial validation accuracy: {}".format(history['val_acc'][0]))
    print("total training time: {}".format(np.sum(times)))
    print("total training time: {}".format(np.average(times)))

    # serialize model to JSON
    model_json = model.to_json()
    modelname = "m"+str(t)+".json"
    with open(modelname, "w") as json_file:
       json_file.write(model_json)
