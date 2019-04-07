
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



#experiment = 'triplet2'
# False for euclidean nearest neighbor, True for cosine similarity distance
#is_use_cosine_similarity = True
#train_labels_tsv = 'labels/triplet2_train.tsv'
#valid_labels_tsv = 'labels/triplet2_valid.tsv'
#train_embedding_csv = 'embeddings/triplet2_train.csv'
#valid_embedding_csv = 'embeddings/triplet2_valid.csv'


#train_embeddings = np.genfromtxt(train_embedding_csv, delimiter=',')
#valid_embeddings = np.genfromtxt(valid_embedding_csv, delimiter=',')
#train_labels = np.genfromtxt(train_labels_tsv, delimiter='\t')
#valid_labels = np.genfromtxt(valid_labels_tsv, delimiter='\t')


def eval_model(train_embeddings,valid_embeddings,train_labels, valid_labels, experiment ):
    cnt = 0
    labels_list = []
    weight_list = []
    clasee_names = []
    with open('evaluator/classes.csv', 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if cnt == 0:
                print(row)
            if cnt > 0:
                labels_list.append(row[0])
                weight_list.append(row[4])
                clasee_names.append(row[7])
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

    # in the following file classes are represented by numbers
    np.savetxt('evaluator/conf_mat_data/' + experiment + '_data.csv', confusion_mat_data, delimiter=",")

    lines = 'valid \t train\n'
    for k in range(N_valid):
        valid_class = clasee_names[np.int(confusion_mat_data[k, 0])].replace(',', '_')
        train_class = clasee_names[np.int(confusion_mat_data[k, 1])].replace(',', '_')
        lines = lines + '{} \t {}\n'.format(valid_class, train_class)

    with open('evaluator/conf_mat_data/' + experiment + '_labels.csv', "w") as text_file:
        text_file.write(lines)


