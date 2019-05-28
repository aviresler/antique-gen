
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import decomposition
import csv
from sklearn.manifold import TSNE
import time
import seaborn as sns
import pandas as pd


labels = np.genfromtxt('labels/triplet_sample_reg_lr5e-6_valid13_57.8.tsv', delimiter=',')
embeddings = np.genfromtxt('embeddings/triplet_sample_reg_lr5e-6_valid13_57.8.csv', delimiter=',')


cnt = 0
cls_dict = {}
color_dict = {}
with open('../data_loader/classes_top200.csv', 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        if cnt == 0:
            print(row)
        if cnt > 0:
            #site, period = row[1].split('_')
            cls_dict[int(row[0])] = int(row[8])
            color_dict[int(row[8])] = row[3]
        cnt = cnt + 1


np.random.seed(7)
X = embeddings
X = X - np.mean(X,axis=0)

color = np.zeros((X.shape[0]), dtype=np.float)
for i,label in enumerate(labels):
    color[i] = cls_dict[label]

color /= np.max(color)

### 2D PCA ###
pca = decomposition.PCA(n_components=2, whiten=False)
X_ = pca.fit_transform(X)

fig, ax = plt.subplots()
im = ax.scatter(X_[:, 0], X_[:, 1],c=color, cmap=plt.cm.nipy_spectral,
           edgecolor='k')
cbar = fig.colorbar(im, ax=ax)
cbar.ax.set_yticklabels([color_dict[0], color_dict[40], color_dict[80], color_dict[119], color_dict[160], color_dict[199]])
plt.tight_layout()
#plt.savefig('results/PCA_2D_whiten=false.png')
plt.show()

### 2D TSNE
#resucing dimentionality to 50
pca_50 = decomposition.PCA(n_components=50)
pca_result_50 = pca_50.fit_transform(X)
print('Cumulative explained variation for 50 principal components: {}'.format(np.sum(pca_50.explained_variance_ratio_)))
print(pca_result_50.shape)

time_start = time.time()
tsne = TSNE(n_components=2, verbose=0, perplexity=40, n_iter=300)
tsne_pca_results = tsne.fit_transform(pca_result_50)
print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))
print(tsne_pca_results.shape)

#fig, ax = plt.subplots()
df = pd.DataFrame(tsne_pca_results)
df['tsne-pca50-one'] = tsne_pca_results[:,0]
df['tsne-pca50-two'] = tsne_pca_results[:,1]
df['y'] = color

ax1 = plt.subplot()
sns.scatterplot(
    x="tsne-pca50-one", y="tsne-pca50-two",
    hue="y",
    palette=sns.color_palette("hls", 200),
    data=df,
    legend=False,
    alpha=0.8,
    ax=ax1
)

#plt.savefig('TSNE_perplexity=40.png')

df = pd.DataFrame(X_)
df['pca-one'] = X_[:, 0]
df['pca-two'] = X_[:, 1]
df['y'] = color

fig, ax2 = plt.subplots()
sns.scatterplot(
    x="pca-one", y="pca-two",
    hue="y",
    palette=sns.color_palette("hls", 200),
    data=df,
    legend=False,
    alpha=0.8,
    ax=ax2
)


fig3, ax3 = plt.subplots()
im = ax3.scatter(X_[:, 0], X_[:, 1], s = 6, c= color,cmap=plt.cm.nipy_spectral)
ax = plt.gca()
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)

#plt.colorbar(im, label="period")
cbar = fig.colorbar(im, ax=ax)
cbar.ax.set_yticklabels([color_dict[0], color_dict[40], color_dict[80], color_dict[119], color_dict[160], color_dict[199]])
plt.tight_layout()

plt.show()

# #make a legend:
# pws = [0.5, 1, 1.5, 2., 2.5]
# for pw in pws:
#     plt.scatter([], [], s=(pw**2)*60, c="k",label=str(pw))
#
# h, l = plt.gca().get_legend_handles_labels()
# plt.legend(h[1:], l[1:], labelspacing=1.2, title="petal_width", borderpad=1,
#             frameon=True, framealpha=0.6, edgecolor="k", facecolor="w")



#plt.colorbar(fig)


#sns.sc
#sns.plt.show()
#sns.plt.colorbar()

#cbar = plt.colorbar( ax=ax1)
#plt.colorbar(im)
#cbar = fig.colorbar(plt, ax=ax1)




# fig = plt.figure(1, figsize=(4, 3))
# plt.clf()
# ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
#
# #fig = plt.figure(figsize = (8,8))
# #ax = fig.add_subplot(1,1,1)
#
# plt.cla()
# pca = decomposition.PCA(n_components=3, whiten=True)
# pca.fit(X)
# X = pca.transform(X)
#
# print(X.shape)
#
#
# ax.scatter(X[:, 0], X[:, 1],X[:, 2],c=color, cmap=plt.cm.nipy_spectral,
#            edgecolor='k')

#ax.w_xaxis.set_ticklabels([])
#ax.w_yaxis.set_ticklabels([])
#ax.w_zaxis.set_ticklabels([])

#plt.show()

#plt.cla()
# pca = decomposition.PCA(n_components=2, whiten=True)
# pca.fit(X)
# X = pca.transform(X)
#
# fig, ax = plt.subplots()
#
# im = ax.scatter(X[:, 0], X[:, 1],c=color, cmap=plt.cm.nipy_spectral,
#            edgecolor='k')
#
# cbar = fig.colorbar(im, ax=ax)
# cbar.ax.set_yticklabels(['Lower Palaeolithic','Pre-Pottery Neolithic B','Middle Bronze IIB','Hellenistic','Early Byzantine','Ottoman'])
# set the color limits - not necessary here, but good to know how.
#im.set_clim(0.0, 1.0)

#ax.w_xaxis.set_ticklabels([])
#ax.w_yaxis.set_ticklabels([])
#ax.w_zaxis.set_ticklabels([])

plt.show()