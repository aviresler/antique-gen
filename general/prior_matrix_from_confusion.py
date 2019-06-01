import numpy as np
import scipy
import csv
import matplotlib.pyplot as plt
import scipy.ndimage
from scipy import signal


num_of_periods = 53

cnt = 0
priod_dict = {}
priod_dict_num_of_images = {}
class_dict_num_img_reverse = {}
class_dict_num_img = {}
with open('../data_loader/classes_top200.csv', 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        if cnt == 0:
            print(row)
        if cnt > 0:
            priod_dict[int(row[8])] = int(row[5])
            priod_dict_num_of_images[int(row[0])] = int(row[5])
            class_dict_num_img[int(row[0])] = row[1]
            class_dict_num_img_reverse[row[1]] = int(row[0])
        cnt = cnt + 1

# read labeled confusion according to archaeologist opinion
num_of_classes = 200
class_mat = np.zeros((num_of_classes,num_of_classes))
class_mat_num_images = np.zeros((num_of_classes,num_of_classes))
equal_level = 100
ambigous_level = 70
simial_level = 40
cnt = 0
errors_gropus = {}
with open('site_period_errors_aviad.csv', 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        if cnt == 0:
            print(row)
        if cnt > 0:
            indication = int(row[2])
            valid_class = row[0]
            train_class = row[1]
            valid_id = class_dict_num_img_reverse[valid_class]
            train_id = class_dict_num_img_reverse[train_class]
            if indication == 2:
                class_mat_num_images[valid_id,train_id] = ambigous_level
                class_mat_num_images[train_id,valid_id] = ambigous_level
            if indication == 3:
                class_mat_num_images[train_id, valid_id] = simial_level



        cnt = cnt + 1



# generate class matrix from period matrix
for i in range(num_of_classes):
    for j in range(num_of_classes):
        if i == j:
            class_mat[i, j] = equal_level
            class_mat_num_images[i, j] = equal_level

plt.imshow(class_mat, interpolation='nearest', cmap=plt.cm.Blues)
plt.show()

plt.imshow(class_mat_num_images, interpolation='nearest', cmap=plt.cm.Blues)
plt.show()

np.savetxt("class_prior.csv", class_mat, delimiter=",")
np.savetxt("../data_loader/class_prior_num_images.csv", class_mat_num_images, delimiter=",")





