from os.path import join
from os import listdir
from skimage.io import imread
import matplotlib.pyplot as plt
import numpy as np
import keras
import keras.layers as L
from keras.models import Model
from skimage.transform import resize
from keras.callbacks import ModelCheckpoint
from skimage import util
from keras.models import load_model


def train_detector(train_gt, train_img_dir, fast_train=True):
    size = (100, 100, 3)
    
    names = sorted(listdir(train_img_dir))
    names=names[:250]
    ar_points = np.ones((len(names), 28))

    images = []
    noise_images = []

    for i, name in enumerate(names):
        image = imread(join(train_img_dir, name))
        k_height, k_width = size[0] / image.shape[0], size[1] / image.shape[1]
        ar_points[i] = train_gt[name]
        ar_points[i][::2] = ar_points[i][::2] * k_height
        ar_points[i][1::2] = ar_points[i][1::2] * k_width
        image = resize(image, output_shape=(100, 100, 3))
        images.append(image)
        noise_images.append(util.random_noise(image))

    images = np.vstack((images, noise_images))
    ar_points = np.vstack((ar_points, ar_points))
    
    perm = np.array([6, 7, 4, 5, 2, 3, 0, 1, 18, 19, 16, 17, 14, 15, 12, 13, 10, 11, 8, 9, 20, 21, 26, 27, 24, 25, 22, 23])

    flip_images = images[:, :,::-1]
    flip_points = ar_points[:,perm]
    flip_points[:,::2] = 100 - flip_points[:,::2]

    ar_points = np.vstack((ar_points, flip_points))
    images = np.vstack((images, flip_images))
    
    
    input_images = L.Input(shape=size)

    conv1 = L.Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(L.BatchNormalization() (input_images))
    pool1 = L.MaxPool2D(pool_size=(2, 2), strides=2)(conv1)

    conv2 = L.Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(pool1)
    pool2 = L.MaxPool2D(pool_size=(2, 2), strides=2)(conv2)

    conv3 = L.Conv2D(filters=128, kernel_size=(3, 3), activation='relu')(pool2)
    pool3 = L.MaxPool2D(pool_size=(2, 2), strides=2)(conv3)

    conv4 = L.Conv2D(filters=256, kernel_size=(3, 3), activation='relu')(pool3)
    pool4 = L.MaxPool2D(pool_size=(2, 2), strides=2)(conv4)

    flatten = L.Flatten()(pool4)
    dense1 = L.Dense(units=256, activation='relu')(flatten)
    dropout = L.Dropout(0.1)(dense1)
    result = L.Dense(units=28)(dropout)

    model = Model(inputs=input_images, outputs=result)
#     checkpoint = ModelCheckpoint('facepoints_model.hdf5', save_best_only=True)
    model.compile('adam', loss='mse')
    
    if fast_train:
        epochs = 1
    else:
        epochs = 40
        
    model.fit(x=images, y=ar_points.astype(np.float32), epochs=epochs, batch_size=32)

    return model

def detect(model, directory) :
    names = np.array(sorted(listdir(directory)))
    size = (100, 100, 3)
    test_images = []
    test_flipped_images = []
    batch_size = 500
    batch_count = (len(names) + batch_size - 1) // batch_size
    all_prediction = []
    shapes = []
    for i in range(batch_count):
        test_images = []
        end = min(len(names), (i + 1) * batch_size)
        for name in names[i * batch_size: end]:
            image = imread(join(directory, name))
            shapes.append(image.shape)
            image = resize(image, output_shape=(100, 100, 3))
            test_images.append(image)
            
        test_images = np.array(test_images)
        test_flipped_images = test_images[:, :,::-1]
        prediction = model.predict(test_images)
        prediction_flipped = model.predict(test_flipped_images)
        perm = np.array([6, 7, 4, 5, 2, 3, 0, 1, 18, 19, 16, 17, 14, 15, 12, 13, 10, 11, 8, 9, 20, 21, 26, 27, 24, 25, 22, 23])
        prediction_flipped = prediction_flipped[:,perm]
        prediction_flipped[:,::2] = 100 - prediction_flipped[:,::2]
        
        prediction = (prediction + prediction_flipped) / 2
        
        if i == 0:
            all_prediction = prediction
        else:
            all_prediction = np.vstack((all_prediction, prediction))  
    
    for i, shape in enumerate(shapes):
        k_height, k_width = shape[0] / size[0] , shape[1] / size[1]
        all_prediction[i][::2]  = all_prediction[i][::2] * k_height
        all_prediction[i][1::2]  = all_prediction[i][1::2] * k_width
        
    return dict(zip(names, all_prediction))
