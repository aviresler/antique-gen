from base.base_data_loader import BaseDataLoader
import numpy as np
import keras
import os
import random
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from data_loader.preprocess_images import get_random_eraser, preprocess_input

class TripletGenerator(keras.utils.Sequence,):
    'Generates data for Keras'
    def __init__(self, config, datagen_args, is_train= True, n_channels=3):
        'Initialization'
        self.config = config
        self.dim = (config.model.img_height, config.model.img_width)
        self.K = config.data_loader.K
        self.P = config.data_loader.P
        self.keras_datagen = ImageDataGenerator(**datagen_args)
        self.batch_size = self.K*self.P

        if is_train:
            self.data_folder = config.data_loader.data_dir_train
        else:
            self.data_folder = config.data_loader.data_dir_valid

        self.n_channels = n_channels
        self.content = {}
        self.active_images = {}
        self.active_classes = []
        self.num_of_images = []
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(self.num_of_images / self.batch_size))

    def __getitem__(self, index):
        'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))

        # pick P classes
        p_classes = random.sample(self.active_classes, self.P)

        image_files_list = []
        labels_list = []
        # collect K images from the chosen classes
        for cls in p_classes:
            # get active images in class
            cls_path = os.path.join(self.data_folder, cls)
            indices = [i for i, x in enumerate(self.active_images[cls_path]) if x == 1]
            num_of_valid_img_in_cls = len(indices)
            if num_of_valid_img_in_cls >= self.K:
                chosen_ind = random.sample(indices, self.K)
                for ind in chosen_ind:
                    img_file = self.content[cls_path][ind]
                    image_files_list.append(os.path.join(self.data_folder, cls, img_file))
                    labels_list.append(int(cls))
                    # zeroing chosen images
                    self.active_images[cls_path][ind] = 0
                indices = [i for i, x in enumerate(self.active_images[cls_path]) if x == 1]
                if not indices:
                    self.active_classes.remove(cls)
            else:  # there are less than K images
                already_chosen_indices = [i for i, x in enumerate(self.active_images[cls_path]) if x == 0]
                extra_ind = random.sample(already_chosen_indices, self.K - num_of_valid_img_in_cls)

                for ind in indices + extra_ind:
                    img_file = self.content[cls_path][ind]
                    image_files_list.append(os.path.join(self.data_folder, cls, img_file))
                    labels_list.append(int(cls))
                    # zeroing chosen images
                    self.active_images[cls_path][ind] = 0
                self.active_classes.remove(cls)

        # generate data
        for i, file_path in enumerate(image_files_list):
            X[i, ] = self.read_and_preprocess_images(file_path)

        return X, np.array(labels_list, dtype=np.int)

    def on_epoch_end(self):
        'initialize indicators arrays '
        img_cnt = 0
        for root, dirs, files in os.walk(self.data_folder):
            if dirs.__len__() != 0:
                self.active_classes = dirs
            for subdir in dirs:
                self.content[os.path.join(root, subdir)] = []
            self.content[root] = files
            self.active_images[root] = [1] * len(files)
            img_cnt += len(files)

        del self.active_images[self.data_folder]
        del self.content[self.data_folder]
        self.num_of_images = img_cnt


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




