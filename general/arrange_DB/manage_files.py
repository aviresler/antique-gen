import shutil
from shutil import copyfile
import csv
import numpy as np
from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential, Model
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras import backend as K
from keras.activations import softmax
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping
from keras.objectives import categorical_crossentropy
import os
import re
import glob



# def copyDirectory(src, dest):
#     try:
#         shutil.copytree(src, dest)
#     # Directories are the same
#     except shutil.Error as e:
#         print('Directory not copied. Error: %s' % e)
#     # Any error saying that the directory doesn't exist
#     except OSError as e:
#         print('Directory not copied. Error: %s' % e)
#


cnt = 0
labels_list = []
weight_list = []
with open('classes.csv', 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        print(row)
        if cnt > 0:

            DIR = '../../data_loader/data/antiques_site_period_all/' + row[0]
            file_list = [name for name in os.listdir(DIR) if (os.path.isfile(os.path.join(DIR, name)) and ('_0' in name))]
            L = len([name for name in os.listdir(DIR) if (os.path.isfile(os.path.join(DIR, name)) and ('_0' in name))])
            artifacts_set = set([name for name in os.listdir(DIR) if ('_0' in name)])
            #print(artifacts_set)
            #print(L)

            # create random artifact mask
            mask = np.random.choice([0, 1], size=(L,), p=[1. / 5, 4. / 5])
            mask = np.ones((L,), dtype=np.int)
            ind = (L-1)*np.random.rand(np.int(np.round(L*0.2)),)
            mask[np.round(ind).astype(int)] = 0

            #print(mask)

            np.testing.assert_almost_equal(0, 1)


            for k in range(L):
                artifact = artifacts_set.pop()
                # find relevant images of the artifacts in the two lists
                matches1 = (x for x in items1 if x.startswith(artifact))
                matches2 = (x for x in items2 if x.startswith(artifact))
                for match in matches1:
                    if mask[k] == 0:
                        copyfile('classes_sorted_train_old/{}/{}'.format(row[0],match), 'classes_sorted_valid/{}/{}'.format(row[0],match))
                    else:
                        copyfile('classes_sorted_train_old/{}/{}'.format(row[0], match),'classes_sorted_train/{}/{}'.format(row[0], match))


            np.testing.assert_almost_equal(0, 1)



        cnt = cnt + 1
