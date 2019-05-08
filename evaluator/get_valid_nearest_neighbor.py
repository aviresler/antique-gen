
import numpy as np
import csv
from sklearn.metrics.pairwise import cosine_similarity

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



# experiment = 'triplet2_sites'
# #False for euclidean nearest neighbor, True for cosine similarity distance
# is_use_cosine_similarity = True
# train_labels_tsv = 'labels/triplet_all_smaller200_lr_5e-6_train_29_acc_56.1.tsv'
# valid_labels_tsv = 'labels/triplet_all_smaller200_lr_5e-6_valid_29_acc_56.1.tsv'
# train_embedding_csv = 'embeddings/triplet_all_smaller200_lr_5e-6_train_29_acc_56.1.csv'
# valid_embedding_csv = 'embeddings/triplet_all_smaller200_lr_5e-6_valid_29_acc_56.1.csv'
#
#
# train_embeddings = np.genfromtxt(train_embedding_csv, delimiter=',')
# valid_embeddings = np.genfromtxt(valid_embedding_csv, delimiter=',')
# train_labels = np.genfromtxt(train_labels_tsv, delimiter='\t')
# valid_labels = np.genfromtxt(valid_labels_tsv, delimiter='\t')
#
# train_labels_mod = np.zeros_like(train_labels,dtype=np.int32)
# valid_labels_mod = np.zeros_like(valid_labels,dtype=np.int32)

# cnt = 0
# cls_dict = {}
# with open('../data_loader/classes_top200.csv', 'r') as f:
#     reader = csv.reader(f)
#     for row in reader:
#         if cnt == 0:
#             print(row)
#         if cnt > 0:
#             #site, period = row[1].split('_')
#             cls_dict[int(row[0])] = int(row[6])
#         cnt = cnt + 1
#
# for i,val_label in enumerate(valid_labels):
#     valid_labels_mod[i] = cls_dict[val_label]
#
# for i,tr_label in enumerate(train_labels):
#     train_labels_mod[i] = cls_dict[tr_label]



def eval_model(train_embeddings,valid_embeddings,train_labels, valid_labels, experiment, is_save_files = True ):
    cnt = 0
    labels_list = []
    #clasee_names = []
    clasee_names = {}
    if is_save_files:
        with open('../data_loader/classes_top200.csv', 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                if cnt == 0:
                    print(row)
                if cnt > 0:
                    clasee_names[int(row[6])] = row[4]
                    #labels_list.append(row[0])
                    #clasee_names.append(row[1])
                cnt = cnt + 1

    similaity_mat = cosine_similarity(valid_embeddings, train_embeddings, dense_output=True)
    max_ind = np.argmax(similaity_mat, axis=1)

    vec0 = valid_labels
    vec1 = train_labels[max_ind]

    num_no_zero = np.count_nonzero(vec0-vec1)
    equal_elements = vec0.shape[0] - num_no_zero
    accuracy = 100*equal_elements/vec0.shape[0]

    print('accuracy= {}'.format(accuracy))

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
            lines = lines + '{} \t {}\n'.format(valid_class, train_class)
        #evaluator /
        with open('conf_mat_data/' + experiment + '_labels.csv', "w") as text_file:
            text_file.write(lines)

    return accuracy
#eval_model(train_embeddings,valid_embeddings,train_labels_mod, valid_labels_mod, 'final_sites', is_save_files = True )
