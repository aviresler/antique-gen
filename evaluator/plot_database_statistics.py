import csv
import numpy as np
import matplotlib.pyplot as plt

images = {}
artifacts = {}
cnt = 0
with open('../data_loader/classes.csv', 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        if cnt > 0:
            images[int(row[0])] = int(row[2])
            artifacts[int(row[8])] = int(row[7])
        cnt = cnt + 1


# plot top 200 images and artifacts bar plot
tops = [200, 556]
x_axis= np.arange(len(images))
y_images = [images[x] for x in x_axis]
y_artifacts = [artifacts[x] for x in x_axis]

# for top in tops:
#     fig, ax = plt.subplots()
#     x = x_axis[:top]
#     y_img = y_images[:top]
#     y_art = y_artifacts[:top]
#     plt.bar(x, y_img,align='edge', alpha=0.75)
#     plt.bar(x, y_art, align='edge', alpha=0.75)
#     #plt.plot( y_img)
#     #plt.plot( y_art)
#     plt.ylabel('number of items', fontsize=14)
#     plt.xlabel('class index', fontsize=14)
#     plt.legend(('images','artifacts'), fontsize=14)
#     plt.grid()
#     plt.savefig('database_' + str(top) )
#
#     #plt.title('Programming language usage')
#     plt.show()

fig, ax = plt.subplots()
x = x_axis[200:557]
y_img = y_images[200:557]
y_art = y_artifacts[200:557]
plt.bar(x, y_img,align='edge', alpha=0.75)
plt.bar(x, y_art, align='edge', alpha=0.75)
#plt.plot( y_img)
#plt.plot( y_art)
plt.ylabel('number of items', fontsize=14)
plt.xlabel('class index', fontsize=14)
plt.legend(('images','artifacts'), fontsize=14)
plt.grid()
plt.show()
plt.savefig('database_200_556')
