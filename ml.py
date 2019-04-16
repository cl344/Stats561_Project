from keras.preprocessing.image import ImageDataGenerator 
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D,ZeroPadding2D, GlobalMaxPooling2D
import keras
from keras.models import model_from_json
import numpy
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
#from lsuv_init import LSUVinit


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






train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

train_generator = train_datagen.flow_from_directory('train_small/',
                                                 target_size = (224, 224),
                                                 color_mode = "rgb",
                                                 batch_size = 32,
                                                 class_mode = 'categorical',
                                                 shuffle = True,
                                                 seed = 2019);

valid_generator = test_datagen.flow_from_directory('valid_small/',
                                            target_size = (224, 224),
                                            color_mode = "rgb",
                                            batch_size = 32,
                                            class_mode = 'categorical',
                                            shuffle = True,
                                            seed = 2019);

STEP_SIZE_TRAIN = train_generator.n//train_generator.batch_size;
STEP_SIZE_VALID = valid_generator.n//valid_generator.batch_size;

model.fit_generator(train_generator,
                         steps_per_epoch = STEP_SIZE_TRAIN,
                         epochs = 20,
                         validation_data = valid_generator,
                         validation_steps = STEP_SIZE_VALID);
                         

			   

# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")

















