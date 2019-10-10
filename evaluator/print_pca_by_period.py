
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import decomposition
import csv
from sklearn.manifold import TSNE
import time
import seaborn as sns
import pandas as pd
import itertools
import matplotlib.cm as cm
import os
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits import mplot3d
from sklearn.random_projection import GaussianRandomProjection


def plot_periods_pca_tsne(embeddings, labels):
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
    im = ax3.scatter(tsne_pca_results[:, 0], tsne_pca_results[:, 1], s = 6, c= color,cmap=plt.cm.nipy_spectral)
    ax = plt.gca()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    #plt.colorbar(im, label="period")
    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.set_yticklabels([color_dict[0], color_dict[40], color_dict[80], color_dict[119], color_dict[160], color_dict[199]])
    plt.tight_layout()

    plt.show()

def plot_sites_in_periods_pca_tsne_old(embeddings, labels, type):
    out_dir = 'results/' + type
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)
    np.random.seed(7)
    # read csv to pandas
    df = pd.read_csv('../data_loader/classes_top200.csv')
    unique_periods_groups = np.unique(df['period_group_rough'])
    for period_group in unique_periods_groups:
        # extract all the labels that belong to this period_group
        sliced_df = df[df['period_group_rough'] ==  period_group]
        group_labels = sliced_df['id']
        inds = []
        colors = []
        for m,label in enumerate(group_labels):
            ind = np.where(labels == label)
            inds.append(ind[0])
            colors.append(ind[0] - ind[0] + m)
            #plot_label = df[df['id'] ==  label]
            #plot_label = plot_label['class_name'].iloc[0]
        inds = [item for sublist in inds for item in sublist]
        colors = [item for sublist in colors for item in sublist]
        group_embeddings = embeddings[inds,:]
        group_labels = labels[inds]
        X = group_embeddings - np.mean(group_embeddings, axis=0)
        colors /= np.max(colors)


        if type.startswith('pca'): ### 2D PCA ###
            pca = decomposition.PCA(n_components=2, whiten=False)
            X_ = pca.fit_transform(X)
        else:
            # reducing dimension to 50
            pca_50 = decomposition.PCA(n_components=30)
            pca_result_50 = pca_50.fit_transform(X)
            time_start = time.time()
            tsne = TSNE(n_components=2, verbose=0, perplexity=40, n_iter=1000)
            X_ = tsne.fit_transform(pca_result_50)
            print('t-SNE done! Time elapsed: {} seconds'.format(time.time() - time_start))


        fig, ax = plt.subplots(figsize=(12, 6))


        uniqe_labels_in_group = np.unique(group_labels)
        colors = cm.rainbow(np.linspace(0, 1, len(uniqe_labels_in_group)))
        for m, label in enumerate(uniqe_labels_in_group):
            ind = np.where(group_labels == label)
            plot_label = df[df['id'] ==  label]
            plot_label = plot_label['class_name'].iloc[0]
            if len(plot_label) > 50:
                plot_label = plot_label[:50]

            ax.scatter(X_[ind[0], 0], X_[ind[0], 1], s=35, label=plot_label,
                        c=np.expand_dims(colors[m], axis=0))

        if ( uniqe_labels_in_group.shape[0] > 30 ):
            size = 5.5
        else:
            size = 8
        ax.legend(loc='upper right', bbox_to_anchor=(1.5, 1),prop={'size': size})
        ax.grid(True)
        plt.title(type + ', group '  + str(period_group))
        plt.tight_layout()
        plt.savefig(out_dir + '/group_' + str(period_group) + '.png')
        plt.show()


def plot_sites_in_periods_pca_tsne(train_embeddings, valid_embeddings, train_labels, valid_labels, type, dim = 2):
    out_dir = 'results/' + type
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)
    np.random.seed(7)

    # concat train and valid embeddings
    embeddings_concat = np.concatenate( (train_embeddings,valid_embeddings), axis=0)
    labels_concat = np.concatenate((train_labels, valid_labels), axis=0)

    data_sets = ['train', 'valid']

    # read csv to pandas
    df = pd.read_csv('../data_loader/classes_top200.csv')
    unique_periods_groups = np.unique(df['period_group_rough'])

    for period_group in unique_periods_groups:
        # extract all the labels that belong to this period_group
        sliced_df = df[df['period_group_rough'] ==  period_group]
        group_labels = sliced_df['id']
        inds = []

        for m,label in enumerate(group_labels):
            ind = np.where(labels_concat == label)
            inds.append(ind[0])

        inds = [item for sublist in inds for item in sublist]
        # flag will indicate
        train_valid_flags = []
        for ind in inds:
            if ind < train_embeddings.shape[0]:
                train_valid_flags.append(1)
            else:
                train_valid_flags.append(0)

        group_embeddings = embeddings_concat[inds,:]
        group_labels = labels_concat[inds]
        X = group_embeddings - np.mean(group_embeddings, axis=0)
        inds_train = np.where(np.array(train_valid_flags) == 1)
        inds_vaild = np.where(np.array(train_valid_flags) == 0)
        labels_train = group_labels[inds_train[0]]
        labels_valid = group_labels[inds_vaild[0]]

        if type.startswith('pca'):  # 2D PCA
            pca = decomposition.PCA(n_components=dim, whiten=False)
            X_ = pca.fit_transform(X)
            X_train = X_[inds_train[0],:]
            X_valid = X_[inds_vaild[0], :]

        else:
            # reducing dimension to 50
            pca_50 = decomposition.PCA(n_components=50)
            pca_result_50 = pca_50.fit_transform(X)
            time_start = time.time()
            tsne = TSNE(n_components=dim, verbose=0, perplexity=40, n_iter=400)
            X_ = tsne.fit_transform(pca_result_50)
            X_train = X_[inds_train[0],:]
            X_valid = X_[inds_vaild[0], :]
            print('t-SNE done! Time elapsed: {} seconds'.format(time.time() - time_start))

        uniqe_labels_in_group = np.unique(group_labels)
        colors = cm.rainbow(np.linspace(0, 1, len(uniqe_labels_in_group)))

        for data_set in data_sets:  # for train and valid
            if dim == 2:
                fig, ax = plt.subplots(figsize=(12, 6))
            else:
                fig = plt.figure(figsize=(12, 6))
                ax = plt.axes(projection="3d")
            for m, label in enumerate(uniqe_labels_in_group):
                if data_set == 'train':
                    X_2_or_3_d = X_train
                    ind = np.where(labels_train == label)
                else:
                    X_2_or_3_d = X_valid
                    ind = np.where(labels_valid == label)

                # plot stle and labels
                plot_label = df[df['id'] ==  label]
                plot_label = plot_label['class_name'].iloc[0]
                if len(plot_label) > 50:
                    plot_label = plot_label[:50]
                if X_2_or_3_d.shape[0] > 600:
                    spot_size = 15
                else:
                    spot_size = 35

                if dim == 2:
                    ax.scatter(X_2_or_3_d[ind[0], 0], X_2_or_3_d[ind[0], 1], s=spot_size, label=plot_label,
                            c=np.expand_dims(colors[m], axis=0))
                else:
                    ax.scatter3D(X_2_or_3_d[ind[0], 0], X_2_or_3_d[ind[0], 1],X_2_or_3_d[ind[0], 2], s=spot_size, label=plot_label,
                               c=np.expand_dims(colors[m], axis=0))

            if (uniqe_labels_in_group.shape[0] > 30):
                size = 5.5
            else:
                size = 8
            ax.legend(loc='upper right', bbox_to_anchor=(1.5, 1),prop={'size': size})
            ax.grid(True)
            plt.title(type + ', ' +   data_set + ' , group '  + str(period_group))
            plt.tight_layout()
            plt.savefig(out_dir + '/group_' + str(period_group) + '_' +  data_set +  '.png')
            plt.show()

def average_embbedings():
    embed_files_train = [ 'efficientNetB3_softmax_f_10__train9_53.99.csv', 'efficientNetB3_softmax_f_4_train_53.4.csv',
                          'efficientNetB3_softmax_6__train13_52.98.csv', 'efficientNetB3_softmax_f_6__train6_52.8.csv',
                           'efficientNetB3_softmax_f_7__train13_52.6.csv', 'efficientNetB3_softmax_f_9__train20_52.6.csv',
                           'efficientNetB3_softmax_f_12__train16_52.29.csv', 'efficientNetB3_softmax_f_1_train_52.1.csv',
                           'efficientNetB3_softmax_5__train9_52.1.csv' , 'efficientNetB3_softmax_8__train15_51.8.csv']

    embed_files_valid = [ 'efficientNetB3_softmax_f_10__valid9_53.99.csv', 'efficientNetB3_softmax_f_4_valid_53.4.csv',
                          'efficientNetB3_softmax_6__valid13_52.98.csv', 'efficientNetB3_softmax_f_6__valid6_52.8.csv',
                           'efficientNetB3_softmax_f_7__valid13_52.6.csv', 'efficientNetB3_softmax_f_9__valid20_52.6.csv',
                           'efficientNetB3_softmax_f_12__valid16_52.29.csv', 'efficientNetB3_softmax_f_1_valid_52.1.csv',
                           'efficientNetB3_softmax_5__valid9_52.1.csv' , 'efficientNetB3_softmax_8__valid15_51.8.csv']

    num_of_runs = [1,5,10]
    for run in num_of_runs:
        print(run)
        embeddings_temp_train = np.genfromtxt('embeddings/' + embed_files_train[0], delimiter=',')
        embeddings_temp_valid = np.genfromtxt('embeddings/' + embed_files_valid[0], delimiter=',')
        #print(embeddings_temp_valid[5, 5])

        average_embeddings_train = np.zeros_like(embeddings_temp_train)
        average_embeddings_valid = np.zeros_like(embeddings_temp_valid)

        for k in range(run):
            print(k)
            train_embed = np.genfromtxt('embeddings/' + embed_files_train[k], delimiter=',')
            valid_embed = np.genfromtxt('embeddings/' + embed_files_valid[k], delimiter=',')
            average_embeddings_train += train_embed
            average_embeddings_valid += valid_embed

        average_embeddings_train = average_embeddings_train / int(run)
        average_embeddings_valid = average_embeddings_valid / int(run)

        #print(average_embeddings_valid[5,5])

        np.savetxt('embeddings/efficientNetB3_softmax_averaged_' + str(run) + '_embeddings_train.csv', average_embeddings_train, delimiter=',')
        np.savetxt('embeddings/efficientNetB3_softmax_averaged_' + str(run) + '_embeddings_valid.csv', average_embeddings_valid, delimiter=',')

def concat_embbedings():
    embed_files_train = [ 'efficientNetB3_softmax_f_10__train9_53.99.csv', 'efficientNetB3_softmax_f_4_train_53.4.csv',
                          'efficientNetB3_softmax_6__train13_52.98.csv', 'efficientNetB3_softmax_f_6__train6_52.8.csv',
                           'efficientNetB3_softmax_f_7__train13_52.6.csv', 'efficientNetB3_softmax_f_9__train20_52.6.csv',
                           'efficientNetB3_softmax_f_12__train16_52.29.csv', 'efficientNetB3_softmax_f_1_train_52.1.csv',
                           'efficientNetB3_softmax_5__train9_52.1.csv' , 'efficientNetB3_softmax_8__train15_51.8.csv']

    embed_files_valid = [ 'efficientNetB3_softmax_f_10__valid9_53.99.csv', 'efficientNetB3_softmax_f_4_valid_53.4.csv',
                          'efficientNetB3_softmax_6__valid13_52.98.csv', 'efficientNetB3_softmax_f_6__valid6_52.8.csv',
                           'efficientNetB3_softmax_f_7__valid13_52.6.csv', 'efficientNetB3_softmax_f_9__valid20_52.6.csv',
                           'efficientNetB3_softmax_f_12__valid16_52.29.csv', 'efficientNetB3_softmax_f_1_valid_52.1.csv',
                           'efficientNetB3_softmax_5__valid9_52.1.csv' , 'efficientNetB3_softmax_8__valid15_51.8.csv']

    embeddings_temp_train = np.genfromtxt('embeddings/' + embed_files_train[0], delimiter=',')
    embeddings_temp_valid = np.genfromtxt('embeddings/' + embed_files_valid[0], delimiter=',')

    num_of_runs = [5]
    for run in num_of_runs:
        print(run)
        concat_embeddings_train = np.zeros((embeddings_temp_train.shape[0],run*embeddings_temp_train.shape[1]), dtype=np.float32 )
        concat_embeddings_valid = np.zeros((embeddings_temp_valid.shape[0],run*embeddings_temp_valid.shape[1]),dtype=np.float32)

        for k in range(run):
            print(k)
            train_embed = np.genfromtxt('embeddings/' + embed_files_train[k], delimiter=',')
            valid_embed = np.genfromtxt('embeddings/' + embed_files_valid[k], delimiter=',')
            concat_embeddings_train[:, k*embeddings_temp_train.shape[1]: (k+1)*embeddings_temp_train.shape[1] ] = train_embed
            concat_embeddings_valid[:, k*embeddings_temp_valid.shape[1]: (k+1)*embeddings_temp_valid.shape[1]] = valid_embed


        if run == 5:
            transformer = GaussianRandomProjection(1500)
            concat_embed = np.concatenate((concat_embeddings_train, concat_embeddings_valid), axis=0)
            X_new = transformer.fit_transform(concat_embed)
            train_embed_new = X_new[:concat_embeddings_train.shape[0], :]
            valid_embed_new = X_new[concat_embeddings_train.shape[0]:, :]
            np.savetxt('embeddings/efficientNetB3_softmax_concat_embeddings_' + str(run) + '_rp1500_train.csv', train_embed_new, delimiter=',')
            np.savetxt('embeddings/efficientNetB3_softmax_concat_embeddings_' + str(run) + '_rp1500_valid.csv', valid_embed_new, delimiter=',')
        else:
            np.savetxt('embeddings/efficientNetB3_softmax_concat_embeddings_' + str(run) + '_train.csv', concat_embeddings_train, delimiter=',')
            np.savetxt('embeddings/efficientNetB3_softmax_concat_embeddings_' + str(run) + '_valid.csv', concat_embeddings_valid, delimiter=',')

if __name__ == '__main__':
    concat_embbedings()
    #average_embbedings()

    # train_labels = np.genfromtxt('labels/efficientNetB3_softmax_averaged_embeddings_train.tsv', delimiter=',')
    # valid_labels = np.genfromtxt('labels/efficientNetB3_softmax_averaged_embeddings_valid.tsv', delimiter=',')
    # train_embeddings = np.genfromtxt('embeddings/efficientNetB3_softmax_concat_rp_embeddings500_train.csv',delimiter=',')
    # valid_embeddings = np.genfromtxt('embeddings/efficientNetB3_softmax_concat_rp_embeddings500_valid.csv', delimiter=',')
    #
    # plot_sites_in_periods_pca_tsne(train_embeddings,valid_embeddings, train_labels, valid_labels, 'tsne_concat_random_projection_embed500_same_pca_3d', dim= 3 )