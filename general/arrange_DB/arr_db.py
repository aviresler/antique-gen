
import csv
import numpy as np
from shutil import copyfile
import re
import os


# cnt = 0
# id = {}
# with open('classes.csv', 'r') as f:
#     reader = csv.reader(f)
#     for row in reader:
#         if cnt == 0:
#             print(row)
#         if cnt > 0:
#             id[row[1]] = int(row[0])
#             #print(os.getcwd())
#             os.mkdir('../../data_loader/data/antiques_site_period_200/train/' + row[0])
#             os.mkdir('../../data_loader/data/antiques_site_period_200/valid/' + row[0])
#             if id[row[1]] > 199:
#                 break
#             #np.testing.assert_almost_equal(0, 1)
#         cnt = cnt + 1
#
# np.testing.assert_almost_equal(0, 1)
# get new class list
cnt = 0
site_list = []
period_list = []
class_list = []
with open('artifacts.csv', 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        if cnt == 0:
            print(row)
        if cnt > 0:
            is_valid = row[5]
            is_indicative = row[6]
            if is_valid == '1' and is_indicative == 'yes':
                site = row[1]
                site = site.replace(',','-')
                site_list.append(site)
                period = row[2]
                period = period.replace(',','-')
                period_list.append(period)
                cls = period  + '_' + site
                class_list.append(cls)
                additional_looks = int(row[3])
                #print(id[cls])
                # copy primary image
                #copyfile('../../data_loader/data/antiques_all/' + row[0] + '_0.jpg',
                #         '../../data_loader/data/antiques_site_period_all/' + str(id[cls]) + '/' + row[0] + '_0.jpg')

                # copy addtional looks
                #if additional_looks != 0:
                #    for mm in range(additional_looks):
                #        copyfile('../../data_loader/data/antiques_all/' + row[0] + '_{}.jpg'.format(mm + 1),
                #                 '../../data_loader/data/antiques_site_period_all/' + str(id[cls]) + '/' + row[0] + '_{}.jpg'.format(mm+1))
            #else:
            #    print('invalid')
        cnt = cnt + 1

set_period = set(period_list)
set_sites = set(site_list)
set_class = set(class_list)
img_in_cls = {}
for cls_ in set_class:
    img_in_cls[cls_] = 0


with open('artifacts.csv', 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        if cnt == 0:
            print(row)
        if cnt > 0:
            is_valid = row[5]
            is_indicative = row[6]
            if is_valid == '1' and is_indicative == 'yes':
                # cls = row[4]
                add_look = int(row[3])

                site = row[1]
                site = site.replace(',', '-')
                site_list.append(site)
                period = row[2]
                period = period.replace(',', '-')
                period_list.append(period)
                cls = period + '_' + site
                class_list.append(cls)
                img_in_cls[cls] = img_in_cls[cls] + 1 + int(add_look)
                # print(id[cls])
        cnt = cnt + 1

for key, value in img_in_cls.items():
    period,site = key.split('_')
    print(key + ',' + str(value) + ',' + period + ',' + site)

np.testing.assert_almost_equal(0,1)
period_set = set(period_list)
for prd_ in period_set:
    print(prd_)

site_set = set(site_list)
for sit_ in site_set:
    print(sit_)


num_of_images = {}
# for cls in set_class:
#     num_of_images[cls] = 0


np.testing.assert_almost_equal(0,1)

    #print(cls)

# count how many artifacts in class
# cnt = 0
# site_list = []
# period_list = []
# class_list = []
# with open('artifacts.csv', 'r') as f:
#     reader = csv.reader(f)
#     for row in reader:
#         if cnt == 0:
#             print(row)
#         if cnt > 0:
#             is_valid = row[5]
#             if is_valid == '1':
#                 site = row[1]
#                 site = site.replace(',','-')
#                 period = row[2]
#                 period = period.replace(',','-')
#                 cls = site + '_' + period
#
#                 additional_look = row[3]
#                 num_of_images[cls] = num_of_images[cls] + 1 + int(additional_look)
#         cnt = cnt + 1
#
# for k, v in num_of_images.items():
#     print(k + ',' +  str(v))

#print(set(period_list))
#print(set(site_list))
#print(set(class_list))
