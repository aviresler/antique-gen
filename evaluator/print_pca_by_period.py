
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import decomposition
import csv


labels = np.genfromtxt('labels/triplet_all_smaller200_lr_5e-6_valid_29_acc_56.1.tsv', delimiter=',')
embeddings = np.genfromtxt('embeddings/triplet_all_smaller200_lr_5e-6_valid_29_acc_56.1.csv', delimiter=',')

cnt = 0
cls_dict = {}
with open('../data_loader/classes_top200.csv', 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        if cnt == 0:
            print(row)
        if cnt > 0:
            site, period = row[1].split('_')
            cls_dict[int(row[0])] = int(row[5])
        if cnt > 201:
            break
        cnt = cnt + 1



np.random.seed(5)

centers = [[1, 1], [-1, -1], [1, -1]]
X = embeddings
color = np.zeros((X.shape[0]), dtype=np.float)
for i,label in enumerate(labels):
    color[i] = cls_dict[label]

color /= np.max(color)

fig = plt.figure(1, figsize=(4, 3))
plt.clf()
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

#fig = plt.figure(figsize = (8,8))
#ax = fig.add_subplot(1,1,1)

plt.cla()
pca = decomposition.PCA(n_components=3, whiten=True)
pca.fit(X)
X = pca.transform(X)

print(X.shape)

# for name, label in [('Setosa', 0), ('Versicolour', 1), ('Virginica', 2)]:
#     ax.text3D(X[y == label, 0].mean(),
#               X[y == label, 1].mean() + 1.5,
#               X[y == label, 2].mean(), name,
#               horizontalalignment='center',
#               bbox=dict(alpha=.5, edgecolor='w', facecolor='w'))
# # Reorder the labels to have colors matching the cluster results
# y = np.choose(y, [1, 2, 0]).astype(np.float)
ax.scatter(X[:, 0], X[:, 1],X[:, 2],c=color, cmap=plt.cm.nipy_spectral,
           edgecolor='k')

#ax.w_xaxis.set_ticklabels([])
#ax.w_yaxis.set_ticklabels([])
#ax.w_zaxis.set_ticklabels([])

plt.show()

#plt.cla()
pca = decomposition.PCA(n_components=2, whiten=True)
pca.fit(X)
X = pca.transform(X)

fig, ax = plt.subplots()

im = ax.scatter(X[:, 0], X[:, 1],c=color, cmap=plt.cm.nipy_spectral,
           edgecolor='k')

cbar = fig.colorbar(im, ax=ax)
cbar.ax.set_yticklabels(['Lower Palaeolithic','Pre-Pottery Neolithic B','Middle Bronze IIB','Hellenistic','Early Byzantine','Ottoman'])
# set the color limits - not necessary here, but good to know how.
#im.set_clim(0.0, 1.0)

#ax.w_xaxis.set_ticklabels([])
#ax.w_yaxis.set_ticklabels([])
#ax.w_zaxis.set_ticklabels([])

plt.show()