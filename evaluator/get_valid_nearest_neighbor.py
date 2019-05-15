
import numpy as np
import csv
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

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





def eval_model_from_csv_files(train_embeddings_csv, valid_embeddings_csv, train_labels_tsv, valid_labels_tsv):

    experiment = 'test'
    classes_csv_file = '../data_loader/classes_top200.csv'

    train_embeddings = np.genfromtxt(train_embeddings_csv, delimiter=',')
    valid_embeddings = np.genfromtxt(valid_embeddings_csv, delimiter=',')
    train_labels = np.genfromtxt(train_labels_tsv, delimiter='\t')
    valid_labels = np.genfromtxt(valid_labels_tsv, delimiter='\t')

    train_labels_period = np.zeros_like(train_labels,dtype=np.int32)
    valid_labels_period = np.zeros_like(valid_labels,dtype=np.int32)
    train_labels_site = np.zeros_like(train_labels,dtype=np.int32)
    valid_labels_site = np.zeros_like(valid_labels,dtype=np.int32)

    cnt = 0
    period_dict = {}
    site_dict = {}
    with open(classes_csv_file, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if cnt > 0:
                site_dict[int(row[0])] = int(row[6])
                period_dict[int(row[0])] = int(row[5])
            cnt = cnt + 1

    for i,val_label in enumerate(valid_labels):
        valid_labels_period[i] = period_dict[val_label]
        valid_labels_site[i] = site_dict[val_label]

    for i,tr_label in enumerate(train_labels):
        train_labels_period[i] = period_dict[tr_label]
        train_labels_site[i] = site_dict[tr_label]

    accuracy = eval_model_topk(train_embeddings, valid_embeddings, train_labels, valid_labels, experiment + '_site_period',
                          is_save_files=True, classes_csv_file=classes_csv_file, class_mode='site_period',is_save_pair_images=True)

    print('accuracy_site_period top_1_3_5= {0:.3f}, {1:.3f}, {2:.3f}'.format(accuracy[0],accuracy[1],accuracy[2]))

    accuracy_period = eval_model_topk(train_embeddings, valid_embeddings, train_labels_period, valid_labels_period, experiment + '_period',
                          is_save_files=True, classes_csv_file=classes_csv_file, class_mode='period')

    print('accuracy_period_top_1_3_5= {0:.3f}, {1:.3f}, {2:.3f}'.format(accuracy_period[0],accuracy_period[1],accuracy_period[2]))

    accuracy_site = eval_model_topk(train_embeddings, valid_embeddings, train_labels_site, valid_labels_site, experiment + '_site',
                          is_save_files=True, classes_csv_file=classes_csv_file, class_mode='site')

    print('accuracy_site top_1_3_5= {0:.3f}, {1:.3f}, {2:.3f}'.format(accuracy_site[0],accuracy_site[1],accuracy_site[2]))


def eval_model_topk(train_embeddings,valid_embeddings,train_labels, valid_labels, experiment, is_save_files = True,
                    classes_csv_file = '',class_mode = 'site_period', is_save_pair_images = False ):
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
        if is_save_pair_images:
            print(k)
            img_valid = mpimg.imread('../data_loader/data/site_period_top_200/valid/' + valid_files[k])
            img_train = mpimg.imread('../data_loader/data/site_period_top_200/train/' + train_files[arg_sort_similaity[k,0]])
            f, (ax1, ax2) = plt.subplots(1, 2)
            ax1.get_xaxis().set_visible(False)
            ax1.get_yaxis().set_visible(False)
            ax2.get_xaxis().set_visible(False)
            ax2.get_yaxis().set_visible(False)

            ax1.imshow(img_valid)
            period, site = clasee_names[int(valid_labels[k])].split('_')
            if ( len(site) > 20):
                site = site[:20]
            valid_title = 'valid\n' + str(int(valid_labels[k])) + '\n' + period + '\n' + site
            ax1.set_title(valid_title)
            ax2.imshow(img_train)
            period, site = clasee_names[neighbours_mat[k,0]].split('_')
            if ( len(site) > 20):
                site = site[:20]
            train_title = 'train\n' + str(int(neighbours_mat[k,0])) + '\n' + period + '\n' + site
            ax2.set_title(train_title)

            #plt.tight_layout()
            #plt.show()

            #plt.tight_layout()
            if neighbours_mat[k,0] == int(valid_labels[k]):
                plt.savefig('results/pairs/correct_' + str(k) +  '_'+ clasee_names[int(valid_labels[k])] + '_' + clasee_names[neighbours_mat[k,0]] + '.png')
            else:
                plt.savefig('results/pairs/wrong_' + str(k) + '_'+ clasee_names[int(valid_labels[k])] + '_' + clasee_names[neighbours_mat[k, 0]] + '.png')

            #np.testing.assert_almost_equal(0,1)




    confusion_mat_data = np.zeros((valid_embeddings.shape[0],N_neighbours+1), dtype=np.int)
    confusion_mat_data[:, 0] = np.squeeze(valid_labels)
    confusion_mat_data[:, 1:] = neighbours_mat


    if is_save_files:
        # in the following file classes are represented by numbersevaluator/
        np.savetxt('conf_mat_data/' + experiment + '_data.csv', confusion_mat_data, delimiter=",")

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

    top_k = [1,3,5]
    indicator_mat = np.zeros((valid_embeddings.shape[0],len(top_k)), dtype=np.int32)
    for i,k in enumerate(top_k):
        for m in range(valid_embeddings.shape[0]):
            label = int(valid_labels[m])
            predictions = confusion_mat_data[m,1:k+1]
            indicator_mat[m,i] = (label in predictions)

    accuracy = 100*np.sum(indicator_mat,axis=0)/valid_embeddings.shape[0]

    return accuracy

def eval_model(train_embeddings,valid_embeddings,train_labels, valid_labels, experiment, is_save_files = True, classes_csv_file = '',class_mode = 'site_period' ):
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
            lines = lines + '{}\t{}\n'.format(valid_class, train_class)
        #evaluator /
        with open('conf_mat_data/' + experiment + '_labels.csv', "w") as text_file:
            text_file.write(lines)

    return accuracy

if __name__ == '__main__':
    eval_model_from_csv_files('embeddings/triplet_sample_reg_lr5e-6_train13_57.8.csv',
                              'embeddings/triplet_sample_reg_lr5e-6_valid13_57.8.csv',
                              'labels/triplet_sample_reg_lr5e-6_train13_57.8.tsv',
                              'labels/triplet_sample_reg_lr5e-6_valid13_57.8.tsv',
                              )

