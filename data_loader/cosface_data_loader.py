from base.base_data_loader import BaseDataLoader
import numpy as np
import keras
import os
import random
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from data_loader.preprocess_images import get_random_eraser, preprocess_input
import re
import csv

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

        self.period_dict = {}
        self.site_dict = {}
        cnt = 0
        with open(self.config.data_loader.classes_info_csv_file, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                if cnt > 0:
                    self.period_dict[int(row[0])] = int(row[5])
                    self.site_dict[int(row[0])] = int(row[6])
                cnt = cnt + 1

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
        priod_label_list = []
        site_label_list = []

        # generate data
        for i, file_path in enumerate(images_path):
            self.images_path_list.remove(file_path)
            X[i, ] = self.read_and_preprocess_images(file_path)

            match = re.search('\/(\d*)_(\d*)\/\d*_\d*.jpg', file_path)
            label_ = int(match.group(1))
            labels_list.append(label_)
            priod_label_list.append(self.period_dict[label_])
            site_label_list.append(self.site_dict[label_])


        # custom loss in keras requires the labels and predictions to be in the same size
        labels = np.zeros((len(labels_list), int(self.config.model.embedding_dim)),dtype=np.int32)
        labels[:,0] = np.array(labels_list, dtype=np.int32)
        labels[:, 1] = np.array(priod_label_list, dtype=np.int32)
        labels[:, 2] = np.array(site_label_list, dtype=np.int32)

        #temp_labels = np.zeros((len(labels_list), 1),dtype=np.int32)
        #temp_labels[:,0] = np.array(labels_list, dtype=np.int32)
        #labels = np.tile(temp_labels, (1, int(self.config.model.embedding_dim)))


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




