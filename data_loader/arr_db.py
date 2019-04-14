
import csv
import numpy as np
import re
import os
from shutil import copyfile

def create_directories():
    if not os.path.exists('data/site_period_all'):
        os.mkdir('data/site_period_all')

    if not os.path.exists('site_period_top_200'):
        os.mkdir('data/site_period_top_200')
        os.mkdir('data/site_period_top_200/train')
        os.mkdir('data/site_period_top_200/valid')

    # create empty directories according to classes
    cnt = 0
    id = {}
    with open('classes.csv', 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if cnt == 0:
                print(row)
            if cnt > 0:
                id[row[1]] = int(row[0])
                direc = 'data/site_period_all/' + row[0]
                if not os.path.exists(direc):
                    os.mkdir(direc)
            cnt = cnt + 1



# copy artifact images to classes folders
def copy_images_to_calses_folders():
    cnt = 0
    with open('artifacts.csv', 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if cnt == 0:
                print(row)
            if cnt > 0:
                is_valid = row[5]
                if is_valid == '1':
                    cls = row[4]
                    add_look = int(row[3])
                    copyfile('data/antiques_all_images/{}_0.jpg'.format(row[0]),
                             'data/site_period_all/{}/{}_0.jpg'.format(cls, row[0]))

                    for k in range(add_look):
                        copyfile('data/antiques_all_images/{}_{}.jpg'.format(row[0], k+1),
                                 'data/site_period_all/{}/{}_{}.jpg'.format(cls, row[0], k+1))

                else:
                    print('invalid')

            cnt = cnt + 1

def split_into_train_valid_top_200():
    cnt = 0
    with open('classes.csv', 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if cnt == 0:
                print(row)
            if cnt > 0:
                cls = row[0]
                # create class folder in train/ valid folder
                if not os.path.exists('data/site_period_top_200/train/{}'.format(cls)):
                    os.mkdir('data/site_period_top_200/train/{}'.format(cls))

                if not os.path.exists('data/site_period_top_200/valid/{}'.format(cls)):
                    os.mkdir('data/site_period_top_200/valid/{}'.format(cls))

                # count how many artifacts in class folder
                items = os.listdir('data/site_period_all/{}'.format(cls))

                # count how many artifacts in the total images
                artifacts = []
                for item in items:
                    match = re.search('(\d*)_(\d*).jpg', item)
                    artifacts.append(match.group(1))
                artifacts_set = set(artifacts)
                L = len(artifacts_set)


                # create random artifact mask
                mask = np.ones((L,), dtype=np.int)
                ind = (L - 1) * np.random.rand(np.int(np.round(L * 0.2)), )
                mask[np.round(ind).astype(int)] = 0

                for k in range(L):
                    artifact = artifacts_set.pop()
                    # find relevant images of the artifacts in the two lists
                    matches = [x for x in items if x.startswith(artifact)]
                    for match in matches:
                        if mask[k] == 1:
                            copyfile('data/site_period_all/{}/{}'.format(cls, match),
                                     'data/site_period_top_200/train/{}/{}'.format(cls, match))
                        else:
                            copyfile('data/site_period_all/{}/{}'.format(cls, match),
                                     'data/site_period_top_200/valid/{}/{}'.format(cls, match))

            if cnt > 199:
                break

            cnt = cnt + 1

create_directories()
copy_images_to_calses_folders()
split_into_train_valid_top_200()
#set_period = set(period_list)
#set_sites = set(site_list)
#set_class = set(class_list)
#num_of_images = {}
#for cls in set_class:
#    num_of_images[cls] = 0
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
