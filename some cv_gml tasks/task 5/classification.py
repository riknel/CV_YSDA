from keras.preprocessing.image import ImageDataGenerator
import keras, keras.layers as L
import keras.applications as zoo
from keras import optimizers
from skimage.transform import resize
from os import listdir
from os.path import join
from skimage.io import imread
from keras.models import Model
import numpy as np
from keras.models import load_model

def train_classifier(train_gt, train_img_dir, fast_train=True):
    if fast_train:
        epochs = 1
    else:
        epochs = 40

    names = sorted(listdir(train_img_dir))
    names = names[:100]
    input_shape = (299, 299, 3)
    images = []
    y = []
    for i, name in enumerate(names):
        image = imread(join(train_img_dir, name))
        y.append(train_gt[name])
        image = zoo.inception_v3.preprocess_input(
            resize(image, output_shape=input_shape, mode='reflect')[None] * 255).reshape(299, 299, 3)
        images.append(image)

    images = np.array(images)
    train_datagen = ImageDataGenerator(
        rotation_range=30,
        zoom_range=0.2,
        horizontal_flip=True)

    train_datagen.fit(images, augment=True)
    X_aug = train_datagen.flow(images, y, batch_size=16)
    model = zoo.InceptionV3(include_top=False, weights='imagenet')

    count_classes = 50
    avg = keras.layers.GlobalAveragePooling2D()(model.layers[-1].output)
    dropout = L.Dropout(0.1)(avg)
    new_output = L.Dense(count_classes, activation='softmax')(dropout)

    new_model = Model(inputs=[model.layers[0].input], output=[new_output])

    for l in new_model.layers[:-200]:
        l.trainable = False

    opt = optimizers.Adam(lr=0.0001)
    new_model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    new_model.fit_generator(X_aug, epochs=epochs, steps_per_epoch=len(images) // 16)
    return new_model


def classify(model, directory):
    names = np.array(sorted(listdir(directory)))

    batch_size = 125
    batch_count = (len(names) + batch_size - 1) // batch_size

    all_prediction = []
    input_shape = (299, 299, 3)

    for i in range(batch_count):
        test_images = []
        end = min(len(names), (i + 1) * batch_size)
        for name in names[i * batch_size: end]:
            image = imread(join(directory, name))
            image = zoo.inception_v3.preprocess_input(resize(image, output_shape=input_shape, mode='reflect')[None] * 255).reshape(299, 299, 3)
            test_images.append(image)

        test_images = np.array(test_images)
        test_flipped_images = test_images[:, :,::-1]

        prediction = model.predict(test_images)
        prediction_flipped = model.predict(test_flipped_images)

        prediction = (prediction + prediction_flipped) / 2
        prediction = np.argmax(prediction, axis=-1)

        if i == 0:
            all_prediction = prediction
        else:
            all_prediction = np.hstack((all_prediction, prediction))


    return dict(zip(names, all_prediction))

