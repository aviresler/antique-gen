
import csv
import wget
import requests
import numpy as np
import re
import os

#change none and to ruler period
# cnt = 0
# is_valid_list = []
# with open('artifacts.csv', 'r') as f:
#     reader = csv.reader(f)
#     for row in reader:
#         if cnt == 0:
#             print(row)
#         if cnt > 0:
#             is_valid = row[5]
#             if is_valid == '1':
#                 period = row[2]
#                 if period == 'none':
#                     # go to artifact web page and get ruler if there is one.
#                     res = requests.get('http://www.antiquities.org.il/t/item_en.aspx?CurrentPageKey={}'.format(row[0]))
#                     # serach for Ruler
#                     m = re.search('Ruler[ ]?: <\/b>[ ]?([^<]*)<b', res.text)
#                     if m is not None:
#                         str1111 = m.group(1)
#                         str1111 = str1111.replace('&nbsp', '')
#                         print(row[0] + ',' + str1111 )
#                     else:
#                         print(row[0] + ',' + 'invalid')
#                     #print(res.text)
#         cnt = cnt + 1

#change none and to ruler period
# cnt = 0
# is_valid_list = []
# with open('artifacts.csv', 'r') as f:
#     reader = csv.reader(f)
#     for row in reader:
#         if cnt == 0:
#             print(row)
#         if cnt > 0:
#             is_valid = row[5]
#             if is_valid == '1':
#                 period = row[2]
#                 if period == 'none':
#                     # go to artifact web page and get ruler if there is one.
#                     res = requests.get('http://www.antiquities.org.il/t/item_en.aspx?CurrentPageKey={}'.format(row[0]))
#                     # serach for Ruler
#                     m = re.search('Ruler[ ]?: <\/b>[ ]?([^<]*)<b', res.text)
#                     if m is not None:
#                         str1111 = m.group(1)
#                         str1111 = str1111.replace('&nbsp', '')
#                         print(row[0] + ',' + str1111 )
#                     else:
#                         print(row[0] + ',' + 'invalid')
#                     #print(res.text)
#         cnt = cnt + 1

cnt = 0
id = {}
with open('classes.csv', 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        if cnt == 0:
            print(row)
        if cnt > 0:
            id[row[1]] = int(row[0])
            os.mkdir('antiques_site_period_all/' + row[0])
        cnt = cnt + 1

np.testing.assert_almost_equal(0,1)
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
            if is_valid == '1':
                site = row[1]
                site = site.replace(',','-')
                site_list.append(site)
                period = row[2]
                period = period.replace(',','-')
                period_list.append(period)
                cls = site + '_' + period
                class_list.append(cls)
                print(id[cls])

            else:
                print('invalid')




        cnt = cnt + 1

set_period = set(period_list)
set_sites = set(site_list)
set_class = set(class_list)
num_of_images = {}
for cls in set_class:
    num_of_images[cls] = 0
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
