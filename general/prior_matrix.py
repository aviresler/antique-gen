import numpy as np
import scipy
import csv
import matplotlib.pyplot as plt
import scipy.ndimage
from scipy import signal


num_of_periods = 53
periods_mat = np.diag(np.ones(num_of_periods))
blurred_periods_mat = np.zeros_like(periods_mat)

data_path = 'period_sigma.csv'
with open(data_path, 'r') as f:
    reader = csv.reader(f, delimiter=',')
    # get header from first row
    headers = next(reader)
    # get all the rows as a list
    data = list(reader)

    # transform data into numpy array
    sigma_vec = np.squeeze( np.array(data).astype(float))

for k in range(num_of_periods):
    orig_row = periods_mat[k,:]
    gauss = signal.gaussian(num_of_periods, std=sigma_vec[k])
    #blurred_row = scipy.ndimage.filters.gaussian_filter1d(orig_row, sigma_vec[k])
    blurred_row = np.convolve(orig_row,gauss, mode='same')
    if k == 1:
        plt.plot(blurred_row)
        plt.show()
    blurred_periods_mat[k,:] = np.around(100*blurred_row)

print(blurred_periods_mat)
plt.imshow(blurred_periods_mat, interpolation='nearest', cmap=plt.cm.Blues)
plt.show()

np.savetxt("periods_prior.csv", blurred_periods_mat, delimiter=",")

cnt = 0
priod_dict = {}
priod_dict_num_of_images = {}
with open('../data_loader/classes_top200.csv', 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        if cnt == 0:
            print(row)
        if cnt > 0:
            priod_dict[int(row[8])] = int(row[5])
            priod_dict_num_of_images[int(row[0])] = int(row[5])
        cnt = cnt + 1

# generate class matrix from period matrix
num_of_classes = 200
class_mat = np.zeros((num_of_classes,num_of_classes))
class_mat_num_images = np.zeros((num_of_classes,num_of_classes))
for i in range(num_of_classes):
    for j in range(num_of_classes):
        class_mat[i,j] = periods_mat[ priod_dict[i], priod_dict[j]]
        class_mat_num_images[i, j] = periods_mat[priod_dict_num_of_images[i], priod_dict_num_of_images[j]]
        if i == j:
            class_mat[i, j] *= 1.3
            class_mat_num_images[i, j] *= 1.3

plt.imshow(class_mat, interpolation='nearest', cmap=plt.cm.Blues)
plt.show()

plt.imshow(class_mat_num_images, interpolation='nearest', cmap=plt.cm.Blues)
plt.show()

np.savetxt("class_prior.csv", class_mat, delimiter=",")
np.savetxt("../data_loader/class_prior_num_images.csv", class_mat_num_images, delimiter=",")





