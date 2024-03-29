
import numpy as np
import csv
from sklearn.metrics.pairwise import cosine_similarity, pairwise_distances
from sklearn import random_projection
from sklearn.random_projection import GaussianRandomProjection
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd
import matplotlib.gridspec as gridspec
import os
import random

def frame_image(img, frame_width):
    b = frame_width # border size in pixel
    ny, nx = img.shape[0], img.shape[1] # resolution / number of pixels in x and y
    if img.ndim == 3: # rgb or rgba array
        framed_img = np.zeros((b+ny+b, b+nx+b, img.shape[2]))
    elif img.ndim == 2: # grayscale image
        framed_img = np.zeros((b+ny+b, b+nx+b))
    framed_img[b:-b, b:-b] = img
    return framed_img

def findCosineSimilarity(source_representation, test_representation):
    a = np.matmul(np.transpose(source_representation), test_representation)
    b = np.sum(np.multiply(source_representation, source_representation))
    c = np.sum(np.multiply(test_representation, test_representation))
    return (a / (np.sqrt(b) * np.sqrt(c)))


def findEuclideanDistance(source_representation, test_representation):
    euclidean_distance = source_representation - test_representation
    euclidean_distance = np.sum(np.multiply(euclidean_distance, euclidean_distance))
    euclidean_distance = np.sqrt(euclidean_distance)
    return euclidean_distance


def eval_model_from_csv_files(train_embeddings_csv, valid_embeddings_csv, train_labels_tsv, valid_labels_tsv, experiment):

    classes_csv_file = '../data_loader/classes_top200.csv'

    train_embeddings = np.genfromtxt(train_embeddings_csv, delimiter=',')
    valid_embeddings = np.genfromtxt(valid_embeddings_csv, delimiter=',')
    train_labels = np.genfromtxt(train_labels_tsv, delimiter='\t')
    valid_labels = np.genfromtxt(valid_labels_tsv, delimiter='\t')

    train_labels_period = np.zeros_like(train_labels,dtype=np.int32)
    valid_labels_period = np.zeros_like(valid_labels,dtype=np.int32)
    train_labels_site = np.zeros_like(train_labels,dtype=np.int32)
    valid_labels_site = np.zeros_like(valid_labels,dtype=np.int32)
    train_labels_id_period = np.zeros_like(train_labels, dtype=np.int32)
    valid_labels__id_period = np.zeros_like(valid_labels, dtype=np.int32)

    cnt = 0
    label_dict = {}
    period_dict = {}
    site_dict = {}
    with open(classes_csv_file, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if cnt > 0:
                site_dict[int(row[0])] = int(row[6])
                period_dict[int(row[0])] = int(row[5])
                label_dict[int(row[0])] = int(row[8])
            cnt = cnt + 1

    for i,val_label in enumerate(valid_labels):
        valid_labels_period[i] = period_dict[val_label]
        valid_labels_site[i] = site_dict[val_label]
        valid_labels__id_period[i] = label_dict[val_label]

    for i,tr_label in enumerate(train_labels):
        train_labels_period[i] = period_dict[tr_label]
        train_labels_site[i] = site_dict[tr_label]
        train_labels_id_period[i] = label_dict[tr_label]

    accuracy = eval_model_topk(train_embeddings, valid_embeddings, train_labels, valid_labels, experiment + '_site_period',
                          is_save_files=False, classes_csv_file=classes_csv_file, class_mode='site_period')
    print('accuracy_site_period top_1_3_5= {0:.3f}, {1:.3f}, {2:.3f}'.format(accuracy[0], accuracy[1], accuracy[2]))


    accuracy_period = eval_model_topk(train_embeddings, valid_embeddings, train_labels_period, valid_labels_period, experiment + '_period',
                          is_save_files=True, classes_csv_file=classes_csv_file, class_mode='period')

    print('accuracy_period_top_1_3_5= {0:.3f}, {1:.3f}, {2:.3f}'.format(accuracy_period[0],accuracy_period[1],accuracy_period[2]))

    accuracy_site = eval_model_topk(train_embeddings, valid_embeddings, train_labels_site, valid_labels_site, experiment + '_site',
                          is_save_files=False, classes_csv_file=classes_csv_file, class_mode='site')

    print('accuracy_site top_1_3_5= {0:.3f}, {1:.3f}, {2:.3f}'.format(accuracy_site[0],accuracy_site[1],accuracy_site[2]))

def eval_model_topk(train_embeddings,valid_embeddings,train_labels, valid_labels, experiment, is_save_files = False,
                    classes_csv_file = '',class_mode = 'site_period', is_save_pair_images = False, is_save_example_images = False):
    cnt = 0
    clasee_names = {}
    rough_period_group_dict = {}
    fine_period_group_dict = {}
    if classes_csv_file == '':
        classes_csv_file = '../data_loader/classes_top200.csv'

    with open(classes_csv_file, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if cnt > 0:
                rough_period_group_dict[int(row[5])] = int(row[9])
                fine_period_group_dict[int(row[5])] = int(row[10])
                if class_mode == 'site_period':
                    clasee_names[int(row[0])] = row[1]
                elif class_mode == 'period':
                    clasee_names[int(row[5])] = row[3]
                elif class_mode == 'site':
                    clasee_names[int(row[6])] = row[4]
                else:
                    raise

            cnt = cnt + 1

    N_neighbours = 50
    neighbours_mat = np.zeros((valid_embeddings.shape[0],N_neighbours),dtype=np.int)

    similaity_mat = cosine_similarity(valid_embeddings, train_embeddings, dense_output=True)

    arg_sort_similaity = np.argsort(similaity_mat, axis=1)
    arg_sort_similaity = np.flip(arg_sort_similaity,axis =1)

    if is_save_pair_images or is_save_example_images:
        valid_files = []
        train_files = []
        with open('train_file_names.csv', 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                train_files.append(row[0])

        with open('valid_file_names.csv', 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                valid_files.append(row[0])


    if is_save_example_images:
        str_ = ''
        #examle_images = [ 229, 525 ,662,1232, 1841]
        examle_images = [386,466,577, 872,1796,  1544]
        fig, axs = plt.subplots(6, 4, figsize=(9, 11))
        #fig, axs = plt.subplots(5, 4)
        #axs = axs.ravel()
        gs1 = gridspec.GridSpec(6, 4)
        gs1.update(wspace=0.025, hspace=0.05)  # set the spacing between axes.

    comm = [125]
    similarity_data = np.zeros((valid_embeddings.shape[0], N_neighbours), dtype=np.float32)
    for kk in range(valid_embeddings.shape[0]):
        k = kk + 0
        neighbours_mat[k,:] = train_labels[arg_sort_similaity[k,:N_neighbours]]
        similarity_data[k,:] = similaity_mat[k, arg_sort_similaity[k,:N_neighbours]]
        if is_save_pair_images and int(valid_labels[k]) in comm :
            print(k)
            fig, axs = plt.subplots(3, 3, figsize=(15, 15))
            axs = axs.ravel()
            #(ax1, ax2, ax3), (ax4, ax5, ax6),(ax7, ax8, ax9) = axs
            img_valid = mpimg.imread('../data_loader/data/data_16_6/site_period_top_200_bg_removed/images_600/valid/' + valid_files[k])
            axs[1].imshow(frame_image(img_valid/255,10), aspect='auto')
            valid_period, valid_site = clasee_names[int(valid_labels[k])].split('_')
            if ( len(valid_site) > 30):
                valid_site = valid_site[:30]
            valid_title = str(int(valid_labels[k])) + ', ' + valid_period + '\n' + valid_site
            axs[1].set_title(valid_title)
            axs[0].axis('off')
            axs[1].axis('off')
            axs[2].axis('off')
            for m in range(6):
                axs[m + 3].axis('off')
                if neighbours_mat[k, m] in comm:
                    img_train = mpimg.imread('../data_loader/data/data_16_6/site_period_top_200_bg_removed/images_600/train/' + train_files[arg_sort_similaity[k,m]])
                    if (neighbours_mat[k, m] != int(valid_labels[k])):
                        axs[m+3].imshow(frame_image(img_train/255,10),aspect='auto')
                    else:
                        axs[m + 3].imshow(img_train, aspect='auto')
                    train_period, train_site = clasee_names[neighbours_mat[k, m]].split('_')
                    if (len(train_site) > 30):
                        train_site = train_site[:30]
                    train_title = str(int(neighbours_mat[k, m])) + ', ' + train_period + '\n' + train_site
                    axs[m+3].set_title(train_title)
                    #axs[m+3].axis('off')

            #plt.show()
            plt.savefig('results/community10/' + valid_period + '_' + valid_site + '_' + str(k) + '.png')
            #assert 0 == 1
            # if neighbours_mat[k,0] == int(valid_labels[k]):
            #     plt.savefig('results/neighbors_6/correct_'+ valid_period + '_' + valid_site + '_' + str(k) + '.png')
            # else:
            #     plt.savefig('results/neighbors_6/wrong_'+ valid_period + '_' + train_period + '_' + valid_site + '_' + train_site + '_' + str(k) +  '.png')
            # plt.clf()

        if is_save_example_images:
            if k in examle_images:
                index = examle_images.index(k)
                img_valid = mpimg.imread('../data_loader/data/data_16_6/site_period_top_200_bg_removed/images_600/valid/' + valid_files[k])
                print(valid_files[k])
                str_ += 'valid,' + clasee_names[int(valid_labels[k])] + '\n'
                ax_valid = plt.subplot(gs1[index*4])
                ax_valid.imshow(img_valid, aspect='auto')
                #ax_valid.xticks([], [])
                #ax_valid.yticks([], [])
                ax_valid.set_xticklabels([])
                ax_valid.set_xticks([])
                ax_valid.set_yticklabels([])
                ax_valid.set_yticks([])
                ax_valid.set_aspect('equal')
                #ax_valid.axis('off')
                for m in range(3):
                    if m == 0:
                        print(train_files[arg_sort_similaity[k, m]])
                    img_train = mpimg.imread('../data_loader/data/data_16_6/site_period_top_200_bg_removed/images_600/train/' + train_files[arg_sort_similaity[k, m]])
                    str_ += 'train,' + clasee_names[neighbours_mat[k, m]] + '\n'
                    ax_train = plt.subplot(gs1[index*4 + m + 1])
                    ax_train.imshow(img_train, aspect='auto')
                    ax_train.axis('off')
                    ax_train.set_aspect('equal')
                str_ += '\n'

    if is_save_example_images:
        with open("examples.csv", "w") as text_file:
            text_file.write(str_)
        print(str_)
        plt.tight_layout()
        plt.savefig('example.png')
        plt.show()
        assert 0 == 1


    confusion_mat_data = np.zeros((valid_embeddings.shape[0],N_neighbours+1), dtype=np.int)
    confusion_mat_data[:, 0] = np.squeeze(valid_labels)
    confusion_mat_data[:, 1:] = neighbours_mat


    if is_save_files:
        # in the following file classes are represented by numbersevaluator/
        np.savetxt('conf_mat_data/' + experiment + '__data.csv', confusion_mat_data, delimiter=",")
        np.savetxt('conf_mat_data/' + experiment + '__similarity_data.csv', similarity_data, delimiter=",")

        lines = 'valid \t train\n'
        for k in range(valid_embeddings.shape[0]):
            valid_class = clasee_names[np.int(confusion_mat_data[k, 0])].replace(',', '_')
            lines = lines + '{}\t'.format(valid_class)
            for m in range(N_neighbours):
                train_class = clasee_names[np.int(confusion_mat_data[k, m+1])].replace(',', '_')
                lines = lines + '{}\t'.format(train_class)
            lines = lines + '\n'

        with open('conf_mat_data/' + experiment + '_labels.csv', "w") as text_file:
            text_file.write(lines)

    top_k = [1, 3, 5]
    if class_mode=='period':
        indicator_mat_rough_periods = np.zeros((valid_embeddings.shape[0], len(top_k)), dtype=np.int32)
        indicator_mat_fine_periods = np.zeros((valid_embeddings.shape[0], len(top_k)), dtype=np.int32)


    indicator_mat = np.zeros((valid_embeddings.shape[0],len(top_k)), dtype=np.int32)
    for i,k in enumerate(top_k):
        for m in range(valid_embeddings.shape[0]):
            label = int(valid_labels[m])
            predictions = confusion_mat_data[m, 1:k + 1]
            indicator_mat[m,i] = (label in predictions)
            if class_mode == 'period':
                lagel_fine_group = fine_period_group_dict[label]
                lagel_rough_group = rough_period_group_dict[label]
                prediction_rough_group = [rough_period_group_dict[x] for x in predictions]
                prediction_fine_group = [fine_period_group_dict[x] for x in predictions]

                indicator_mat_rough_periods[m, i] = (lagel_rough_group in prediction_rough_group)
                indicator_mat_fine_periods[m, i] = (lagel_fine_group in prediction_fine_group)


    accuracy = 100*np.sum(indicator_mat,axis=0)/valid_embeddings.shape[0]
    if class_mode == 'period':
        accuracy_fine = 100 * np.sum(indicator_mat_fine_periods, axis=0) / valid_embeddings.shape[0]
        accuracy_rough = 100 * np.sum(indicator_mat_rough_periods, axis=0) / valid_embeddings.shape[0]
        print('accuracy rough periods')
        print(accuracy_rough)
        print('accuracy fine periods')
        print(accuracy_fine)

    return accuracy


def print_example_images(train_embeddings,valid_embeddings,train_labels, valid_labels, experiment, is_save_files = True,
                    classes_csv_file = '',class_mode = 'site_period', is_save_pair_images = False ):

    example_ind = [91, 223, 431, 439, 572, 585]
    cnt = 0
    clasee_names = {}
    if is_save_files:
        if classes_csv_file == '':
            classes_csv_file = '../data_loader/classes_top200.csv'

        with open(classes_csv_file, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                if cnt > 0:
                    if class_mode == 'site_period':
                        clasee_names[int(row[0])] = row[1]
                    elif class_mode == 'period':
                        clasee_names[int(row[5])] = row[3]
                    elif class_mode == 'site':
                        clasee_names[int(row[6])] = row[4]
                    else:
                        raise

                cnt = cnt + 1

    N_neighbours = 5
    neighbours_mat = np.zeros((valid_embeddings.shape[0],N_neighbours),dtype=np.int)

    similaity_mat = cosine_similarity(valid_embeddings, train_embeddings, dense_output=True)

    arg_sort_similaity = np.argsort(similaity_mat, axis=1)
    arg_sort_similaity = np.flip(arg_sort_similaity,axis =1)

    if is_save_pair_images:
        valid_files = []
        train_files = []
        with open('train_file_names_15_5.csv', 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                train_files.append(row[0])

        with open('valid_file_names_15_5.csv', 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                valid_files.append(row[0])


    for k in range(valid_embeddings.shape[0]):
        neighbours_mat[k,:] = train_labels[arg_sort_similaity[k,:N_neighbours]]

    fig, axes = plt.subplots(2, 6)
    lines = ''

    xlabels = [ '(a)','(b)','(c)','(d)','(e)','(f)']
    for i,ind in enumerate(example_ind):
        img_valid_ = mpimg.imread('../data_loader/data/site_period_top_200/valid/' + valid_files[ind])
        img_train_ = mpimg.imread(
            '../data_loader/data/site_period_top_200/train/' + train_files[arg_sort_similaity[ind, 0]])

        img_train = 255*np.ones((600,600,3), dtype=np.uint8)
        margin = 600-int(img_train_.shape[1])
        start_ind = int(margin/2)

        img_train[:img_train_.shape[0],start_ind:(start_ind+img_train_.shape[1]),:] = img_train_

        img_valid = 255*np.ones((600,600,3), dtype=np.uint8)
        margin = 600-int(img_valid_.shape[1])
        start_ind = int(margin/2)
        img_valid[:img_valid_.shape[0],start_ind:(start_ind+img_valid_.shape[1]),:] = img_valid_

        axes[0, i].imshow(img_valid)
        axes[0, i].axis('off')
        axes[0, i].get_yaxis().set_visible(False)
        axes[0, i].get_xaxis().set_visible(False)
        axes[0, i].set_title( xlabels[i] + '\n')
        period, site = clasee_names[int(valid_labels[ind])].split('_')
        #if ( len(site) > 20):
        #     site = site[:20]
        lines = lines + 'valid,ind,{},period,{},site,{}\n'.format(ind,period,site )
        axes[1, i].imshow(img_train)
        axes[1, i].axis('off')
        axes[1, i].get_yaxis().set_visible(False)
        axes[1, i].get_xaxis().set_visible(False)
        #axes[1, i].set(xlabel= '\n' + xlabels[i])
        period, site = clasee_names[neighbours_mat[ind, 0]].split('_')
        lines = lines + 'train,ind,{},period,{},site,{}\n'.format(ind, period, site)


    plt.savefig('results/choosen_pairs/wrong_0.png')
    with open('results/choosen_pairs/wrong_0.csv', "w") as text_file:
        text_file.write(lines)

    plt.tight_layout()
    plt.show()

    # for k in range(valid_embeddings.shape[0]):
    #     neighbours_mat[k,:] = train_labels[arg_sort_similaity[k,:N_neighbours]]
    #     if is_save_pair_images:
    #         print(k)
    #         img_valid = mpimg.imread('../data_loader/data/site_period_top_200/valid/' + valid_files[k])
    #         img_train = mpimg.imread('../data_loader/data/site_period_top_200/train/' + train_files[arg_sort_similaity[k,0]])
    #
    #         if k in example_ind:
    #             axes[0, cur_ind].imshow(img_valid)
    #             axes[1, cur_ind].imshow(img_train)
    #             cur_ind += 1

            # #check if k is contained in the example images
            # f, (ax1, ax2) = plt.subplots(1, 2)
            #
            # ax1.get_yaxis().set_visible(False)
            # ax2.get_xaxis().set_visible(False)
            # ax2.get_yaxis().set_visible(False)
            #
            # ax1.imshow(img_valid)
            # period, site = clasee_names[int(valid_labels[k])].split('_')
            # if ( len(site) > 20):
            #     site = site[:20]
            # valid_title = 'valid\n' + str(int(valid_labels[k])) + '\n' + period + '\n' + site
            # ax1.set_title(valid_title)
            # ax2.imshow(img_train)
            # period, site = clasee_names[neighbours_mat[k,0]].split('_')
            # if ( len(site) > 20):
            #     site = site[:20]
            # train_title = 'train\n' + str(int(neighbours_mat[k,0])) + '\n' + period + '\n' + site
            # ax2.set_title(train_title)

            #plt.tight_layout()
            #plt.show()

            #plt.tight_layout()
            #if neighbours_mat[k,0] == int(valid_labels[k]):
            #    plt.savefig('results/pairs/correct_' + str(k) +  '_'+ clasee_names[int(valid_labels[k])] + '_' + clasee_names[neighbours_mat[k,0]] + '.png')
            #else:
            #    plt.savefig('results/pairs/wrong_' + str(k) + '_'+ clasee_names[int(valid_labels[k])] + '_' + clasee_names[neighbours_mat[k, 0]] + '.png')


    return 0


def eval_model(train_embeddings,valid_embeddings,train_labels, valid_labels, experiment, is_save_files = False, classes_csv_file = '',class_mode = 'site_period' ):
    cnt = 0
    clasee_names = {}
    if is_save_files:
        if classes_csv_file == '':
            classes_csv_file = 'data_loader/classes_top200.csv'

        with open(classes_csv_file, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                if cnt > 0:
                    if class_mode == 'site_period':
                        clasee_names[int(row[0])] = row[1]
                    elif class_mode == 'period':
                        clasee_names[int(row[5])] = row[3]
                    elif class_mode == 'site':
                        clasee_names[int(row[6])] = row[4]
                    else:
                        print('invalid class mode')
                        raise
                cnt = cnt + 1

    similaity_mat = cosine_similarity(valid_embeddings, train_embeddings, dense_output=True)
    max_ind = np.argmax(similaity_mat, axis=1)

    vec0 = valid_labels
    vec1 = train_labels[max_ind]

    num_no_zero = np.count_nonzero(vec0-vec1)
    equal_elements = vec0.shape[0] - num_no_zero
    accuracy = 100*equal_elements/vec0.shape[0]

    #print('accuracy= {}'.format(accuracy))

    N_valid = valid_embeddings.shape[0]
    confusion_mat_data = np.zeros((N_valid,2), dtype=np.int)

    confusion_mat_data[:, 0] = np.squeeze(vec0)
    confusion_mat_data[:, 1] = np.squeeze(vec1)

    if is_save_files:
        # in the following file classes are represented by numbersevaluator/
        np.savetxt('conf_mat_data/' + experiment + '_data.csv', confusion_mat_data, delimiter=",")

        lines = 'valid \t train\n'
        for k in range(N_valid):
            valid_class = clasee_names[np.int(confusion_mat_data[k, 0])].replace(',', '_')
            train_class = clasee_names[np.int(confusion_mat_data[k, 1])].replace(',', '_')
            lines = lines + '{}\t{}\n'.format(valid_class, train_class)
        #evaluator /
        with open('conf_mat_data/' + experiment + '_labels.csv', "w") as text_file:
            text_file.write(lines)

    return accuracy

def eval_model_per_period_group(train_embeddings,valid_embeddings,train_labels, valid_labels, priod_group_column ):

    similaity_mat = cosine_similarity(valid_embeddings, train_embeddings, dense_output=True)
    max_ind = np.argmax(similaity_mat, axis=1)
    vec0 = valid_labels
    vec1 = train_labels[max_ind]

    # changing labels to fine groups label

    num_no_zero = np.count_nonzero(vec0 - vec1)
    equal_elements = vec0.shape[0] - num_no_zero
    accuracy = 100 * equal_elements / vec0.shape[0]

    df = pd.read_csv('../data_loader/classes_top200.csv')
    period_groups = df[priod_group_column]
    uniqe_period_groups = np.unique(period_groups.values)

    accuracy_per_period = []
    for group in uniqe_period_groups:
        ind = []
        temp = np.where(vec0 == group)
        ind.extend(temp[0])
        # generate list of images files in each group
        # group_slice = df[period_groups == group]
        # periods_group_ids = group_slice['period id'].values
        # unique_group_period_ids = np.unique(periods_group_ids)
        # ind = []
        # for period in unique_group_period_ids:
        #     temp = np.where(vec0 == period)
        #     ind.extend(temp[0])

        valid_per_group = vec0[ind]
        train_per_group = vec1[ind]

        num_no_zero = np.count_nonzero(valid_per_group-train_per_group)
        equal_elements = valid_per_group.shape[0] - num_no_zero
        accuracy_per_period.append(100*equal_elements/valid_per_group.shape[0])

    return accuracy, accuracy_per_period, vec0, vec1


def eval_model_per_period_group_quiz_check(actual_answers,predicitons, priod_group_column ):

    vec0 = actual_answers
    vec1 = predicitons


    num_no_zero = np.count_nonzero(vec0 - vec1)
    equal_elements = vec0.shape[0] - num_no_zero
    accuracy = 100 * equal_elements / vec0.shape[0]

    df = pd.read_csv('../data_loader/classes_top200.csv')
    period_groups = df[priod_group_column]
    uniqe_period_groups = np.unique(period_groups.values)

    accuracy_per_period = []

    print(uniqe_period_groups)
    for group in uniqe_period_groups:
        ind = []
        temp = np.where(vec0 == group)
        ind.extend(temp[0])
        # # generate list of images files in each group
        # group_slice = df[period_groups == group]
        # periods_group_ids = group_slice['period_group_fine'].values
        # unique_group_period_ids = np.unique(periods_group_ids)
        # print(unique_group_period_ids)
        # ind = []
        # for period in unique_group_period_ids:
        #     temp = np.where(vec0 == period)
        #     ind.extend(temp[0])

        valid_per_group = vec0[ind]
        train_per_group = vec1[ind]

        num_no_zero = np.count_nonzero(valid_per_group-train_per_group)
        equal_elements = valid_per_group.shape[0] - num_no_zero
        accuracy_per_period.append(100*equal_elements/valid_per_group.shape[0])

    return accuracy, accuracy_per_period, vec0, vec1

def test_model_on_query_img_csv(train_embeddings_csv, valid_embeddings_csv, train_labels_tsv, valid_labels_tsv):
    train_embeddings = np.genfromtxt(train_embeddings_csv, delimiter=',')
    valid_embeddings = np.genfromtxt(valid_embeddings_csv, delimiter=',')
    train_labels = np.genfromtxt(train_labels_tsv, delimiter='\t')
    valid_labels = np.genfromtxt(valid_labels_tsv, delimiter='\t')

    img_ind = 151
    query_embeddings = valid_embeddings[img_ind,:]
    query_embeddings = query_embeddings[np.newaxis,:]
    query_label = valid_labels[img_ind]

    test_model_on_query_imgages(train_embeddings,train_labels, query_embeddings,query_label,200,
                            classes_csv_file='', class_mode='site_period', isPrint=True)


def test_model_on_query_imgages(train_embeddings, train_labels, query_embeddings, query_label, num_of_classes,
                    classes_csv_file = '',class_mode = 'site_period', isPrint = False):



    N_neighbours = 10
    num_of_sampels = query_embeddings.shape[0]
    probability = np.zeros((num_of_sampels, num_of_classes),dtype=np.float32)

    similaity_mat = cosine_similarity(query_embeddings, train_embeddings, dense_output=True)
    arg_sort_similaity = np.argsort(similaity_mat, axis=1)
    arg_sort_similaity = np.flip(arg_sort_similaity,axis =1)
    neighbours_ind = arg_sort_similaity[:,:N_neighbours]

    for k in range(num_of_sampels):
        neighbours_cls = train_labels[neighbours_ind[k,:]]
        neighbours_similarity = similaity_mat[k,neighbours_ind]

        unique_cls = np.unique(neighbours_cls)

        for cls in unique_cls:
            ind_cls = np.where(neighbours_cls == cls)
            ind_cls = ind_cls[0]
            num_of_neighbours = ind_cls.shape[0]
            cls_similarity_score = np.sum(neighbours_similarity[k,ind_cls])
            probability[k,int(cls)] = cls_similarity_score

        probability[k,:] = probability[k,:]/np.sum(probability[k,:])


        if isPrint:
            pred_string = get_prediction_string(probability[k,:], query_label, classes_csv_file = classes_csv_file,class_mode = class_mode)
            print(pred_string)


    return probability



def get_prediction_string(probability, query_label, classes_csv_file = '',class_mode = 'site_period'):

    cnt = 0
    clasee_names = {}
    if classes_csv_file == '':
        classes_csv_file = '../data_loader/classes_top200.csv'

    with open(classes_csv_file, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if cnt > 0:
                if class_mode == 'site_period':
                    clasee_names[int(row[0])] = row[1]
                elif class_mode == 'period':
                    clasee_names[int(row[5])] = row[3]
                elif class_mode == 'site':
                    clasee_names[int(row[6])] = row[4]
                else:
                    raise
            cnt = cnt + 1

    arg_sort_probability = np.argsort(probability)
    arg_sort_probability = np.flip(arg_sort_probability)
    lines = 'query: \n'
    print(query_label)
    if isinstance(query_label, str):
        temp_str = query_label + '\n'
    else:
        temp_str = str(int(query_label)) + ' ' + clasee_names[query_label] + '\n'

    lines += temp_str
    lines += 'predictions: \n'
    for ind in arg_sort_probability:
        if probability[ ind] > 0:
            period, site = clasee_names[ind].split('_')
            if ( len(site) > 20):
                site = site[:20]
            lines += str(ind) + ' {0:0.3}'.format(probability[ind]) + ' ' +  period  + '_' + site + '\n'

    return lines


def get_class_centroid_embeddings(embeddings, labels):
    unique_labels = np.unique(labels)
    num_of_classes = unique_labels.shape[0]
    mean_embeddings = np.zeros((num_of_classes,embeddings.shape[1]),dtype=np.float32)
    new_labels = np.zeros((num_of_classes, ), dtype=np.int32)
    for k, label in enumerate(unique_labels):
        ind = np.where(labels==label)
        ind = ind[0]
        class_embbed = embeddings[ind,:]
        mean_embeddings[k, :] = np.mean(class_embbed,axis=0)
        new_labels[k] = label

    np.savetxt('labels/efficientNetB3_softmax_concat_rp_embeddings1500_valid_centroids.tsv', new_labels, delimiter=',')
    np.savetxt('embeddings/efficientNetB3_softmax_concat_rp_embeddings1500_valid_centroids.csv', mean_embeddings,delimiter=',')

    return mean_embeddings,new_labels


def random_projection(train_embeddings_csv,valid_embeddings_csv ):
    train_embeddings = np.genfromtxt(train_embeddings_csv, delimiter=',')
    valid_embeddings = np.genfromtxt(valid_embeddings_csv, delimiter=',')
    concat_embed = np.concatenate((train_embeddings, valid_embeddings), axis=0)
    print(train_embeddings.shape)
    print(valid_embeddings.shape)
    print(concat_embed.shape)

    sizes = [1500]
    for size in sizes:
        transformer = GaussianRandomProjection(size)
        X_new = transformer.fit_transform(concat_embed)
        train_embed_new = X_new[:train_embeddings.shape[0],:]
        valid_embed_new = X_new[train_embeddings.shape[0]:, :]

        print(train_embed_new.shape)
        print(valid_embed_new.shape)

        np.savetxt('embeddings/efficientNetB3_softmax_concat10_rp_embeddings' + str(size) +  '_train.csv', train_embed_new,
                   delimiter=',')
        np.savetxt('embeddings/efficientNetB3_softmax_concat10_rp_embeddings'  + str(size) + '_valid.csv', valid_embed_new,
                   delimiter=',')


def get_similarity_matrix():

    classes_csv_file = '../data_loader/classes_top200.csv'
    cnt = 0
    class_id_dict = {}
    with open(classes_csv_file, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if cnt > 0:
                class_id_dict[int(row[0])] = int(row[8])
            cnt = cnt + 1

    data_sets = ['train', 'valid']
    embedding_csv_files = ['embeddings/efficientNetB3_softmax_concat_rp_embeddings1500_train_centroids.csv',\
                           'embeddings/efficientNetB3_softmax_concat_rp_embeddings1500_valid_centroids.csv']
    labels_csv_files = ['labels/efficientNetB3_softmax_concat_rp_embeddings1500_train_centroids.tsv',\
                           'labels/efficientNetB3_softmax_concat_rp_embeddings1500_valid_centroids.tsv']

    for m,data_set in enumerate(data_sets):
        embeddings = np.genfromtxt(embedding_csv_files[m], delimiter=',')
        labels = np.genfromtxt(labels_csv_files[m], delimiter=',')
        new_sorted_by_period_embeddings = np.zeros_like(embeddings)
        for k in range(embeddings.shape[0]):
            old_label = int(labels[k])
            new_label = class_id_dict[old_label]
            new_sorted_by_period_embeddings[new_label] = embeddings[old_label]
        #similaity_mat = cosine_similarity(new_sorted_by_period_embeddings, dense_output=True)
        similaity_mat = pairwise_distances(new_sorted_by_period_embeddings)
        #similaity_mat = similaity_mat.clip(min=0)
        #similaity_mat = np.absolute(similaity_mat)
        np.savetxt('similarity_mat/efficientNetB3_softmax_concat_rp_embeddings1500_centroids_clip_dis_' + data_sets[m] + '.csv',similaity_mat ,
                   delimiter=',')


        #similaity_mat = similaity_mat.astype('float') / similaity_mat.sum(axis=1)[:, np.newaxis]
        plt.imshow(similaity_mat, interpolation='nearest')
        plt.title('pairwise distance matrix ' + data_sets[m])
        plt.colorbar()

        #plt.ylabel('True label')
        #plt.xlabel('Predicted label')
        plt.tight_layout()
        plt.savefig('similarity_mat/similarity_mat_dis_' + data_sets[m] + '.png')
        plt.show()


def get_similarity_summary():
    classes_csv_file = '../data_loader/classes_top200.csv'
    cnt = 0
    class_name_dict = {}
    with open(classes_csv_file, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if cnt > 0:
                class_name_dict[int(row[8])] = row[1]
            cnt = cnt + 1

    similariy_mat = np.genfromtxt('similarity_mat/efficientNetB3_softmax_concat_rp_embeddings1500_centroids_clip_dis_valid.csv', delimiter=',')

    N = 10
    lines = ''
    for k in range(similariy_mat.shape[0]):
        vec = similariy_mat[k,:]
        #ind_max = np.flip(np.argsort(vec), axis=0)
        ind_max = np.argsort(vec)
        ind_max = ind_max[1:N]

        scores = vec[ind_max]
        scores = np.trim_zeros(scores, 'b')

        period, site = class_name_dict[k].split('_')

        lines = lines + 'true_id, {}, true_site, {}, true_period, {}\n'.format(k, site, period)

        for m in range(scores.shape[0]):
            pred_id = ind_max[m]
            period, site = class_name_dict[pred_id].split('_')
            temp_str = 'pred_id, {}, pred_site, {}, pred_period, {}, pred_score, {:0.3f}\n'.format(pred_id, site, period, scores[m])
            lines = lines + temp_str

        lines = lines + '\n'

    with open('similarity_mat/summary_pair_dist_valid.csv', "w") as text_file:
        text_file.write(lines)







if __name__ == '__main__':
    train_embeddings_csv = 'embeddings/efficientNetB3_softmax_concat_embeddings_10_rp1500_train.csv'
    valid_embeddings_csv = 'embeddings/efficientNetB3_softmax_concat_embeddings_10_rp1500_valid.csv'
    train_labesl_tsv = 'labels/efficientNetB3_softmax_averaged_embeddings_train.tsv'
    valid_labesl_tsv = 'labels/efficientNetB3_softmax_averaged_embeddings_valid.tsv'

    #random_projection(train_embeddings_csv,valid_embeddings_csv)

    eval_model_from_csv_files(train_embeddings_csv,valid_embeddings_csv,train_labesl_tsv,valid_labesl_tsv, 'efficientNetB3_conc10_rp_full_neighbor')