from base.base_data_loader import BaseDataLoader
import numpy as np
import keras
import os
import random
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from data_loader.preprocess_images import get_random_eraser, preprocess_input
import re

class CosFaceGenerator(keras.utils.Sequence,):
    'Generates data for Keras'
    def __init__(self, config, datagen_args, is_train= True, n_channels=3):
        'Initialization'
        self.config = config
        self.dim = (config.model.img_height, config.model.img_width)
        self.keras_datagen = ImageDataGenerator(**datagen_args)
        self.batch_size = self.config.data_loader.batch_size
        self.num_of_classes = []

        if is_train:
            self.data_folder = config.data_loader.data_dir_train
        else:
            self.data_folder = config.data_loader.data_dir_valid

        self.n_channels = n_channels
        self.images_path_list = []
        self.num_of_images = []
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(self.num_of_images / self.batch_size))

    def __getitem__(self, index):
        'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)


        # choose batch_size images and remove them from image list
        try:
            images_path = random.sample(self.images_path_list, self.batch_size)
        except:
            images_path = random.sample(self.images_path_list, len(self.images_path_list))
            self.on_epoch_end()
            print('resetting')

        # Initialization
        X = np.empty((len(images_path), *self.dim, self.n_channels),dtype=np.float32)
        labels_list = []

        # generate data
        for i, file_path in enumerate(images_path):
            self.images_path_list.remove(file_path)
            X[i, ] = self.read_and_preprocess_images(file_path)

            match = re.search('\/(\d*)\/\d*_\d*.jpg', file_path)
            labels_list.append(int(match.group(1)))


        # custom loss in keras requires the labels and predictions to be in the same size
        temp_labels = np.zeros((len(labels_list), 1),dtype=np.int32)
        temp_labels[:,0] = np.array(labels_list, dtype=np.int32)
        labels = np.tile(temp_labels, (1, int(self.config.model.embedding_dim)))


        return X, labels


    def on_epoch_end(self):
        'initialize indicators arrays '
        self.images_path_list = []
        # r=root, d=directories, f = files
        for r, d, f in os.walk(self.data_folder):
            for file in f:
                if '.jpg' in file:
                    self.images_path_list.append(os.path.join(r, file))

        random.shuffle(self.images_path_list)

        self.num_of_images = len(self.images_path_list)



    def read_and_preprocess_images(self, image_path):
        # read image from disk
        img = image.load_img(image_path, target_size=(self.dim[0], self.dim[1],3))
        img = image.img_to_array(img)

        if self.config.data_loader.is_use_cutOut:
            preprocess_function = preprocess_input
        else:
            preprocess_function = get_random_eraser(v_l=0, v_h=1)

        img = preprocess_function(img)

        # random transform
        x = self.keras_datagen.random_transform(img)

        return x




