import wget
import requests
import re
import csv
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt

sits = []
periods = []
class_list = []
count = 0

with open("info/classes", "r") as ins:
    array = []
    for line in ins:
        class_list.append(line)

#print(class_list)
add_look_list = []
classes = []
classes_with_add_look = []
with open('info/summary.csv', 'r') as csvFile:
    reader = csv.reader(csvFile)
    for row in reader:
        if count > 0:
            #print(row[3])
            add_look_list.append(row[3])
            classes.append(row[4])
            classes_with_add_look.append(row[4])
            for k in range(np.int(row[3])):
                classes_with_add_look.append(row[4])

            #sits.append(row[1].replace('\n', ' '))
            #periods.append(row[2].replace('\n', ' '))
            #cls = '{}_{}\n'.format(row[1],row[2])
            #print(class_list.index(cls)+1)
            # find class number for each item
            #class_list.append(cls)

        count = count + 1

csvFile.close()

artifacts = np.asarray(classes, dtype=np.int32)
full_data = np.asarray(classes_with_add_look, dtype=np.int32)

print(artifacts.shape)
print(full_data.shape)



# art = np.loadtxt('info/artifact_in_class_order')
# img = np.loadtxt('info/images_in_class')
#
# plt.figure()
# plt.bar(range(1,1263,1), art)
# plt.title("artifacts in class")
# plt.xlabel('class, sorted by num of artifacts')
# plt.ylabel('num of artifacts')
# plt.show()
#
# plt.figure()
# plt.bar(range(1,1263,1), img)
# plt.title("images in class")
# plt.xlabel('class, sorted by num of images')
# plt.ylabel('num of images')
# plt.show()




# plt.figure()
# n, bins, patches = plt.hist(artifacts, range(1,1264,1))
# for k in range(len(n)):
#     print(bins[k])
#     print(n[k])
#
# with open('info/artifcats_in_class.txt', 'a') as the_file:
#  for k in range(len(n)):
#      the_file.write('{}\n'.format(n[k]))
#
#
# #plt.hist(artifacts, bins=7930)
# #plt.plot(aaa)
# plt.title("artifacts in class - histogram")
# plt.xlabel('class')
# plt.ylabel('num of artifacts')
# plt.show()
#
# plt.figure()
# n, bins, patches = plt.hist(full_data, range(1,1264,1))
# for k in range(len(n)):
#     print(bins[k])
#     print(n[k])
#
# with open('info/images_in_class.txt', 'a') as the_file:
#  for k in range(len(n)):
#      the_file.write('{}\n'.format(n[k]))
# print(bins)
# #plt.hist(artifacts, bins=7930)
# #plt.plot(aaa)
# plt.title("images in class - histogram")
# plt.xlabel('class')
# plt.ylabel('num of images')
# plt.show()



set_sites = set(sits)
set_periods = set(periods)
set_class = set(class_list)

site_len = len(set_sites)
period_len = len(set_periods)
clase_len = len(set_class)
# with open('info/unique_sits.csv', 'w') as myfile:
#     wr = csv.writer(myfile, delimiter=',')
#     print('fsfcs')
#     wr.writerow(set_sites)
#     #for k in range(site_len):
#     #    site = set_sites.pop()
#     #    print(site)
#     #    wr.writerow(site)
#         #print(site)
#
# with open('info/unique_periods.csv', 'w') as myfile:
#     wr = csv.writer(myfile, delimiter=',')
#     print('fsfcs')
#     wr.writerow(set_periods)
#     #for k in range(period_len):
#     #    period = set_periods.pop()
#     #    print(period)
#     #    wr.writerow(period)
#     #    #print(site)
#
# with open('info/unique_classes.txt', 'a') as the_file:
#     for k in range(clase_len):
#         clss = set_class.pop()
#         print(clss)
#         the_file.write('{}\n'.format(clss))
#         #wr.write(clss)
#     #the_file.write('Hello\n')
#
# with open('info/unique_classes.csv', 'w') as myfile:
#     wr = csv.writer(myfile, delimiter=',')
#     print('fsfcs')
#     #wr.writerow(set_class)

        #wr.write(clss)
    #    #print(site)

print(site_len)
print(period_len)
print(clase_len)
# with open('info/unique_periods.csv', 'wb') as myfile:
#     wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
#     for period in set_periods:
#         wr.writerow(period)
#         print(period)


#os.rename('a.txt', 'b.kml')

# count  = 0
# file_list = os.listdir('antiques')
# additional_looks = np.zeros((7930,1), dtype=np.int32)
#
# for filename in os.listdir('antiques'):
#     if filename.endswith(".jpg"):
#         m = re.search('(.*)_(.).jpg', filename)
#         if m.group(2) is not '0':
#             index = np.int(m.group(1)) -1
#             print(index)
#             additional_looks[index,0] = additional_looks[index,0] + 1
#             #print(filename)
#             #print(m.group(1))
#             #print(m.group(2))
# np.savetxt("additionl_looks.csv", additional_looks, delimiter=",")
#     # m = re.search('.*_.*jpg', filename)
    # #print(m)
    # if m is None:
    #     if filename.endswith(".txt"):
    #         continue
    #     m = re.search('(.*)\.jpg', filename)
    #     print(filename)
    #     #os.rename(os.path.join('antiques', filename), os.path.join('antiques', m.group(1) + '_0.jpg' ))
    #     print(m.group(1))
    #     print(filename)

   # if filename.endswith(".jpg") or filename.endswith(".jpg"):
   #     print(os.path.join('antiques', filename))
   #     continue
   # else:
   #     continue

# offset = 7750
# num_of_images = 7930-offset+1


#num_of_images = 10

#num_of_images = 2000

# res = requests.get('http://www.antiquities.org.il/t/item_en.aspx?CurrentPageKey=1')
# text_file = open("Output111.txt", "w")
# text_file.write(res.text)
# text_file.close()
#
# res = requests.get('http://www.antiquities.org.il/t/item_en.aspx?CurrentPageKey=37')
# text_file = open("Output37.txt", "w")
# text_file.write(res.text)
# text_file.close()

# res = requests.get('http://www.antiquities.org.il/t/item_en.aspx?CurrentPageKey=15')
# m = re.search('href="(images[^ ]*jpg)"', res.text)
# print(m.group(1))
# text_file = open("Output2.txt", "w")
# text_file.write(res.text)
# text_file.close()


# is_additional_look = []
#
# for k in range(num_of_images):
#     res = requests.get('http://www.antiquities.org.il/t/item_en.aspx?CurrentPageKey={}'.format(offset+k))
#     # get image link
#     m = re.findall('Additional Looks', res.text)
#     if not not m:    # list is not empty
#         print(k + offset)
#         m = re.findall('href="([^ ]*pic_id[^ ]*)"', res.text)
#         set_image = set(m)
#         for kk in range(len(set_image)):
#             res = requests.get('http://www.antiquities.org.il/t/{}'.format(set_image.pop()))
#             m = re.search('href="(images[^ ]*jpg)"', res.text)
#             print(m.group(1))
#             wget.download('http://www.antiquities.org.il/t/{}'.format(m.group(1)),
#                           out='antiques_additional_looks/{}_{}.jpg'.format((offset + k), kk+1))
#             #index.append(offset + k)
#     else:            # list is empty
#         print(-1)
#
#
#
