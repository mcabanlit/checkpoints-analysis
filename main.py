from matplotlib import pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import  img_to_array,  load_img
import scipy

datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='constant')  #Try other fill modes, e.g. nearest, reflect, wrap

############################
#Single image augmentation for demonstration purposes
img = load_img('data/cell_images/test/Parasitized/C33P1thinF_IMG_20150619_114756a_cell_179.png')
# uses Pillow in the backend, so need to convert to array

x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)

# the .flow() command below generates batches of randomly transformed images
# and saves the results to the `preview/` directory
i = 0
for batch in datagen.flow(x, batch_size=1,
                          save_to_dir='data/augmented', save_prefix='aug', save_format='png'):
    i += 1
    if i > 20:
        break  # otherwise the generator would loop indefinitely

#End Demo of single image
##########################################################


from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
#from keras import backend as K

SIZE = 150
###2 conv and pool layers. with some normalization and drops in between.

INPUT_SHAPE = (SIZE, SIZE, 3)   #change to (SIZE, SIZE, 3)

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=INPUT_SHAPE))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

print(model.summary())
###############################################################
batch_size = 16
#Let's prepare our data. We will use .flow_from_directory()
#to generate batches of image data (and their labels)
#directly from our png in their respective folders.

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=45,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

# this is the augmentation configuration we will use for validation:
# only rescaling. But you can try other operations
validation_datagen = ImageDataGenerator(rescale=1./255)

# this is a generator that will read pictures found in
# subfolers of 'data/train', and indefinitely generate
# batches of augmented image data
train_generator = train_datagen.flow_from_directory(
        'data/cell_images/train',  # this is the input directory
        target_size=(150, 150),  # all images will be resized to 64x64
        batch_size=batch_size,
        class_mode='binary')  # since we use binary_crossentropy loss, we need binary labels

# this is a similar generator, for validation data
validation_generator = validation_datagen.flow_from_directory(
        'data/cell_images/test',
        target_size=(150, 150),
        batch_size=batch_size,
        class_mode='binary')

# training_generator = ImageDataGenerator(rescale=1./255, rotation_range=7, horizontal_flip=True, shear_range=0.2, height_shift_range=0.07, zoom_range=0.2)
# test_generator = ImageDataGenerator(rescale=1./255)
#
# training_base= training_generator.flow_from_directory('data/cell_images/train' , target_size=(150,150),batch_size=1, class_mode='binary')
#
# test_base = test_generator.flow_from_directory('data/cell_images/test', target_size=(150, 150),batch_size=1, class_mode='binary')

# print(training_base.samples, test_base.samples)
#Add Callbacks, e.g. ModelCheckpoints, earlystopping, csvlogger.
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger

#ModelCheckpoint callback saves a model at some interval.
filepath="data/saved_models/weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5" #File name includes epoch and validation accuracy.
#Use Mode = max for accuracy and min for loss.
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

callbacks_list = [checkpoint]

#We can now use these generators to train our model.
#Give this a name so we can call it later for plotting loss, accuracy etc. as a function of epochs.
history = model.fit_generator(
        train_generator,
        # training_base,
        # steps_per_epoch=2000 // batch_size,
        steps_per_epoch=15,
        # steps_per_epoch=2000 // batch_size,    #The 2 slashes division return rounded integer
        epochs=5,
        # validation_data=test_base,
        validation_data=validation_generator,
        validation_steps=800 // batch_size,
        callbacks=callbacks_list)
model.save('malaria_augmented_model.h5')  # always save your weights after training or during training

# evaluating the model
train_loss, train_acc = model.evaluate_generator(train_generator, steps=16)
validation_loss, test_acc = model.evaluate_generator(validation_generator, steps=16)
print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))


# plot training history
print("Values stored in history are ... \n", history.history)
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show()


#####################################################