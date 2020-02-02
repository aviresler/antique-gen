
import numpy as np
import csv
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import networkx as nx
import community
import pandas as pd
import matplotlib.image as mpimg

def convert_id():

    classes_csv_file = '../../data_loader/classes_top200.csv'

    cnt = 0
    period_id_dict = {}
    with open(classes_csv_file, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if cnt > 0:
                period_id_dict[int(row[0])] = int(row[8])
            cnt = cnt + 1


    full_conf = np.genfromtxt('efficientNetB3_conc10_rp_full_neighbor_site_period_data.csv', delimiter=',')
    full_conf_converted = np.zeros_like(full_conf)
    for k in range(full_conf.shape[0]):
        for m in range(full_conf.shape[1]):
            full_conf_converted[k,m] = period_id_dict[full_conf[k,m]]
    np.savetxt('efficientNetB3_conc10_rp_full_neighbor_site_period_data_id_period.csv', full_conf_converted, delimiter=",")



def get_communities(num_of_neighbors, is_self_loops, relevant_period_groups, full_confusion_csv, classes_csv_file, priod_group_column, similarty_csv = ''):
    """generates communities based on modularity maximization

    Args:
        num_of_neighbors (int): number of neighbors to be considered, number between 1-50.
        is_self_loops (bool): Whether to form a graph which has edges between nodes to themselves.
        relevant_period_groups (list of int): period groups that should be considered when forming graph. if -1, all period
        groups. The list of period groups is in classes_csv_file, at priod_group_column.
        full_confusion_csv (str): path to csv file with the confusion data.
        classes_csv_file (str): path to csv file with the classes data.
        priod_group_column (str): relevant column for period_groups in classes_csv_file

    Returns:
        summery string


    """

    # generate class_names dict
    cnt = 0
    class_name_dict = {}
    with open(classes_csv_file, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if cnt > 0:
                class_name_dict[int(row[8])] = row[1]
            cnt = cnt + 1


    full_conf = np.genfromtxt(full_confusion_csv, delimiter=',')
    relevant_conf = full_conf[:,:num_of_neighbors+1]
    flatten_conf = np.zeros((relevant_conf.shape[0]*num_of_neighbors,2), dtype=np.int32)
    if similarty_csv != '':
        similarity_mat = np.genfromtxt(similarty_csv, delimiter=',')
        similarity_conf_mat = np.zeros((200, 200), dtype=np.float32)
        print(similarity_mat.shape)


    row  = 0
    for k in range(relevant_conf.shape[0]):
        for m in range(num_of_neighbors):
            flatten_conf[row, 0] = relevant_conf[k,0]
            flatten_conf[row,1] = relevant_conf[k,m+1]
            if similarty_csv != '':
                similarity_conf_mat[int(relevant_conf[k,0]), int(relevant_conf[k,m+1]) ] += similarity_mat[k, m]

            row = row + 1

    confusion_mat = confusion_matrix(flatten_conf[:,0], flatten_conf[:,1])
    if similarty_csv != '':
        confusion_mat = similarity_conf_mat

    confusion_mat = confusion_mat.astype('float') / confusion_mat.sum(axis=1)[:, np.newaxis]
    symmetric_confusion = (confusion_mat + np.transpose(confusion_mat)) / 2
    if not is_self_loops:
        np.fill_diagonal(symmetric_confusion, 0)

    # taking only the relevant classes
    if relevant_period_groups != -1:
        df = pd.read_csv(classes_csv_file)
        period_groups = df[priod_group_column]
        relevant_classes = []
        for group in relevant_period_groups:
            group_slice = df[period_groups == group]
            relevant_classes.extend(group_slice['id_period_sorted'].values)

        L = len(relevant_classes)
        relevant_confusion = np.zeros((L,L), dtype=np.float32)
        class_node_dict = {}
        for m,cls_i in enumerate(relevant_classes):
            class_node_dict[m] = cls_i
            for n,cls_j in enumerate(relevant_classes):
                relevant_confusion[m,n] = symmetric_confusion[cls_i,cls_j]
    else:
        relevant_confusion = symmetric_confusion

    G = nx.from_numpy_matrix(relevant_confusion)

    # find best communities based on modularity grade
    resolution_vec = np.linspace(0.0,2,50)
    mod_vec = np.zeros_like(resolution_vec)
    best_modularity = -1
    best_communities = -1
    best_res = -1
    for k in range(resolution_vec.size):
        partition = community.best_partition(G, weight='weight', resolution=resolution_vec[k])
        modularity = community.modularity(partition, G, weight='weight')
        mod_vec[k] = modularity
        if (modularity > best_modularity):
            best_modularity = modularity
            best_communities = partition
            best_res = resolution_vec[k]

    summary_str = 'best resolution: %.3f\nbest modularity: %.3f\nnumber of communities: %d' % (best_res,best_modularity,len(set(best_communities.values())))

    #plt.plot(resolution_vec,mod_vec)
    #plt.show()

    # generate community summary file
    count = 0
    strr = ''
    summary_file_name = 'community_summary.csv'
    for com in set(best_communities.values()):
        count += 1.
        list_nodes = [nodes for nodes in best_communities.keys() if best_communities[nodes] == com]
        strr += 'community,' + str(com) + '\n'
        for nd in list_nodes:
            if relevant_period_groups == -1:
                strr += class_name_dict[nd] + ',id,' + str(nd) + '\n'
            else:
                strr += class_name_dict[class_node_dict[nd]] + ',id,' + str(class_node_dict[nd]) + '\n'
        strr += '\n'
    with open(summary_file_name, "w") as text_file:
        text_file.write(strr)

    print(strr)
    # summary for map visualization tool
    strr = ''
    for k in range(relevant_confusion.shape[0]):
         comm = partition[k]
         comm_members = [nodes for nodes in partition.keys() if partition[nodes] == comm]
         if relevant_period_groups == -1:
            strr += 'id,' + str(k) + ',community,' + str(comm) + ',community_members,'
         else:
             strr += 'id,' + str(class_node_dict[k]) + ',community,' + str(comm) + ',community_members,'
         for member in comm_members:
             if relevant_period_groups == -1:
                strr += str(member) + ','
             else:
                 strr += str(class_node_dict[member]) + ','
         strr += '\n'
    with open('nodes_communities.csv', "w") as text_file:
        text_file.write(strr)



    return summary_str

def print_conf_examples( full_confusion_csv, classes_csv_file, community,  choosen_sprite_inde = -1 ):

    images_path = '../../data_loader/data/data_16_6/site_period_top_200_bg_removed/images_600/'
    # generate class_names dict
    cnt = 0
    class_name_dict = {}
    with open(classes_csv_file, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if cnt > 0:
                class_name_dict[int(row[8])] = row[1]
            cnt = cnt + 1

    valid_files = []
    train_files = []
    with open('../train_file_names.csv', 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            train_files.append(row[0])

    with open('../valid_file_names.csv', 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            valid_files.append(row[0])


    full_conf = np.genfromtxt(full_confusion_csv, delimiter=',')
    for k in range(full_conf.shape[0]):
        if ( full_conf[k,0] in community):
            fig, axs = plt.subplots(3, 3, figsize=(15, 15))
            axs = axs.ravel()
            # (ax1, ax2, ax3), (ax4, ax5, ax6),(ax7, ax8, ax9) = axs
            img_valid = mpimg.imread( images_path + 'valid/' + valid_files[k])
            axs[1].imshow(img_valid, aspect='auto')
            valid_period, valid_site = class_name_dict[full_conf[k,0]].split('_')
            if (len(valid_site) > 30):
                valid_site = valid_site[:30]
            valid_title = str(full_conf[k,0]) + ', ' + valid_period + '\n' + valid_site
            axs[1].set_title(valid_title)
            axs[0].axis('off')
            axs[1].axis('off')
            axs[2].axis('off')
            for m in range(1,7):
                if ( full_conf[k,m] in community):
                    img_train = mpimg.imread( images_path + ' train/' + train_files[ arg_sort_similaity[k, m]])
                    axs[m + 3].imshow(img_train, aspect='auto')
                    train_period, train_site = clasee_names[neighbours_mat[k, m]].split('_')
                    if (len(train_site) > 30):
                        train_site = train_site[:30]
                    train_title = str(int(neighbours_mat[k, m])) + ', ' + train_period + '\n' + train_site
                    axs[m + 3].set_title(train_title)
                    axs[m + 3].axis('off')
            assert 0 == 1

    print(full_conf.shape)


if __name__ == '__main__':
    # group of -1 means all classes
    #summary = get_communities(6,True,[1, 2, 3, 4, 5],'efficientNetB3_conc10_rp_full_neighbor_site_period_id_period__data.csv','classes_top200.csv','period_group_rough','efficientNetB3_conc10_rp_full_neighbor_site_period_id_period__similarity_data.csv')
    summary = get_communities(6, True, [1, 2, 3, 4, 5],'efficientNetB3_conc10_rp_full_neighbor_site_period_id_period__data.csv','classes_top200.csv', 'period_group_rough',)

    print(summary)

    #print_conf_examples('efficientNetB3_conc10_rp_full_neighbor_site_period_data_id_period.csv', 'classes_top200.csv', [16,22,33], choosen_sprite_inde=-1)