import os
#import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential, Model
from keras.layers import Dense,Add,Input
from keras.layers import Dropout
from keras.layers import Flatten
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.optimizers import Adam
from keras.layers import Activation
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.initializers import glorot_normal
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
from keras.utils import plot_model
from keras import backend as K
from tensorflow.python.keras.callbacks import CSVLogger
#from IPython.display import SVG
#from keras.utils import model_to_dot
from sklearn.metrics import classification_report, confusion_matrix
if K.backend()=='tensorflow':
    K.common.image_dim_ordering() == 'th'

import tensorflow as tf
import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin'



####### Define your hypermeters


BATCH_SIZE= 8
EPOCHS = 20
data_augmentation = False


#IP_SHAPE = [28,28,1]

################ end #################
def lr_schedule(epoch):
    """Learning Rate Schedule
    Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
    Called automatically every epoch as part of callbacks during training.
    # Arguments
        epoch (int): The number of epochs
    # Returns
        lr (float32): learning rate
    """
    lr = 5e-3
    if epoch > 15:
        lr *= 0.5e-3
    elif epoch > 12:
        lr *= 1e-3
    elif epoch > 9:
        lr *= 1e-2
    elif epoch > 3:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr

class vgg16:
    @staticmethod
    def skipresvgg(X):
        X_shortcut = X
        # X = Conv2D(input_shape=input_shape, filters=64, kernel_size=(3, 3), padding="same")(X)
        # X = Activation('relu')(X)
        X = Conv2D(filters=64, kernel_size=(3, 3), padding="same")(X)
        X = Activation('relu')(X)
        X = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(X)
        X = Conv2D(filters=128, kernel_size=(3, 3), padding="same")(X)
        X = Activation('relu')(X)
        X = Conv2D(filters=128, kernel_size=(3, 3), padding="same")(X)
        X = Activation('relu')(X)
        X = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(X)
        X = Conv2D(filters=256, kernel_size=(3, 3), padding="same")(X)
        X = Activation('relu')(X)
        X = Conv2D(filters=256, kernel_size=(3, 3), padding="same")(X)
        X = Activation('relu')(X)
        X = Conv2D(filters=256, kernel_size=(3, 3), padding="same")(X)
        X = Activation('relu')(X)
        X = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(X)
        X = Conv2D(filters=512, kernel_size=(3, 3), padding="same")(X)
        X = Activation('relu')(X)
        X = Conv2D(filters=512, kernel_size=(3, 3), padding="same")(X)
        X = Activation('relu')(X)
        X = Conv2D(filters=512, kernel_size=(3, 3), padding="same")(X)
        X = Activation('relu')(X)
        X = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(X)
        X = Conv2D(filters=512, kernel_size=(3, 3), padding="same")(X)
        X = Activation('relu')(X)
        X = Conv2D(filters=512, kernel_size=(3, 3), padding="same")(X)
        X = Activation('relu')(X)
        X = Conv2D(filters=512, kernel_size=(3, 3), padding="same")(X)
        X = Activation('relu')(X)
        X = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(X)
        X_shortcut = Conv2D(filters = 512,kernel_size=(1,1), padding = 'valid')(X_shortcut)
        X = Add()([X,X_shortcut])
        X = Activation('relu')(X)
        return X
    def build(input_shape, classes):
        # Defining the model to be used
        ####################### TO DO ######################
        ####### Implement Lenet model    ##########
        # Zero-Padding
        X_input = Input(input_shape)
        X = ZeroPadding2D((3, 3))(X_input)
        # Stage 1
        X = Conv2D(64, (7, 7), strides=(2, 2), name='conv1')(X)
        #X = BatchNormalization(axis=3, name='bn_conv1')(X)
        X = Activation('relu')(X)
        X = MaxPooling2D((3, 3), strides=(2, 2))(X)
        X = vgg16.skipresvgg(X)
        X = vgg16.skipresvgg(X)
        X = vgg16.skipresvgg(X)
        X = Conv2D(64, (1, 1), strides=(1, 1), name='convfinal')(X)
        X = MaxPooling2D((3, 3), strides=(2, 2))(X)
        #X = MaxPooling2D((5, 5), strides=(2, 2))(X)
        X = Flatten()(X)
        X = Dense(units=4096, activation="relu")(X)
        X = Dense(units=4096, activation="relu")(X)
        X = Dense(units=classes, activation="softmax")(X)
        model = Model(inputs = X_input, outputs = X, name='ResVgg')

        return model


X_train = np.load('X_train_saved.npy')
X_test = np.load('X_test_saved.npy')
y_train = np.load('y_train_saved.npy')
y_test = np.load('y_test_saved.npy')
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')
print(X_train.shape)
print(X_test.shape)
print(y_train)
class1_count = 0
class2_count = 0
class3_count = 0
for i in y_train:
    if i== 0:
        class1_count+=1
    if i == 1:
        class2_count+=1
    if i== 2:
        class3_count+=1
print(class1_count)
print(class2_count)
print(class3_count)
# convert class vectors to binary class matrices
y_train = np_utils.to_categorical(y_train, 3)
y_test = np_utils.to_categorical(y_test, 3)
# Input image dimensions.
input_shape = X_train.shape[1:]

model = vgg16.build(input_shape=input_shape, classes=3)
opt = SGD(lr=lr_schedule(0))
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

# Prepare model model saving directory.
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'cifar10_%s_model.{epoch:03d}.h5' # model_type
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
filepath = os.path.join(save_dir, model_name)
print(model.summary())
# plot_model(model, to_file='ResVgg.png')
# SVG(model_to_dot(model).create(prog='dot.exe', format='svg'))
# Prepare callbacks for model saving and for learning rate adjustment.
checkpoint = ModelCheckpoint(filepath=filepath,
                             monitor='val_acc',
                             verbose=1,
                             save_best_only=True)

lr_scheduler = LearningRateScheduler(lr_schedule)

lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                               cooldown=0,
                               patience=5,
                               min_lr=0.5e-6)
csv_logger = CSVLogger('ResVgglog.csv', append=True, separator=';')
callbacks = [checkpoint, lr_reducer, lr_scheduler,csv_logger]
#Run training, with or without data augmentation.
if not data_augmentation:
    print('Not using data augmentation.')
    hist = model.fit(X_train, y_train,
              batch_size=BATCH_SIZE,
              epochs=EPOCHS,
              validation_data=(X_test, y_test),
              shuffle=True,callbacks = callbacks)
else:
    print('Using real-time data augmentation.')
    # This will do preprocessing and realtime data augmentation:
    datagen = ImageDataGenerator(
        # set input mean to 0 over the dataset
        featurewise_center=False,
        # set each sample mean to 0
        samplewise_center=False,
        # divide inputs by std of dataset
        featurewise_std_normalization=False,
        # divide each input by its std
        samplewise_std_normalization=False,
        # apply ZCA whitening
        zca_whitening=False,
        # epsilon for ZCA whitening
        zca_epsilon=1e-06,
        # randomly rotate images in the range (deg 0 to 180)
        rotation_range=0,
        # randomly shift images horizontally
        width_shift_range=0.1,
        # randomly shift images vertically
        height_shift_range=0.1,
        # set range for random shear
        shear_range=0.,
        # set range for random zoom
        zoom_range=0.,
        # set range for random channel shifts
        channel_shift_range=0.,
        # set mode for filling points outside the input boundaries
        fill_mode='nearest',
        # value used for fill_mode = "constant"
        cval=0.,
        # randomly flip images
        horizontal_flip=True,
        # randomly flip images
        vertical_flip=False,
        # set rescaling factor (applied before any other transformation)
        rescale=None,
        # set function that will be applied on each input
        preprocessing_function=None,
        # image data format, either "channels_first" or "channels_last"
        data_format=None
        # fraction of images reserved for validation (strictly between 0 and 1)
       )

    # Compute quantities required for featurewise normalization
    # (std, mean, and principal components if ZCA whitening is applied).
    datagen.fit(X_train)

    # Fit the model on the batches generated by datagen.flow().
    hist = model.fit_generator(datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),steps_per_epoch=X_train.shape[0]//BATCH_SIZE,
                        validation_data=(X_test, y_test),
                        epochs=EPOCHS, verbose=1, workers=4,callbacks = callbacks)


scores = model.evaluate(X_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])
#f1 = plt()
#f1 = plt.axes()
plt.figure(0)
plt.plot(hist.history["acc"])
plt.plot(hist.history['val_acc'])
plt.title("model accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
plt.legend(["Training Accuracy", "Validation Accuracy"])
plt.grid('on')
plt.savefig("ResVggOutputAccuracy.png")
#f2 = plt()
#f2 = plt.axes()
#plt.clf()
plt.figure(1)
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title("model loss")
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.legend(["Training loss", "Validation Loss"])
plt.grid('on')
plt.savefig("ResVggOutputLoss.png")
#history=model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=True, validation_split=0.25)