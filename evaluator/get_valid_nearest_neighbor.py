
import numpy as np
import csv

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

cnt = 0
labels_list = []
weight_list = []
clasee_names = []
with open('classes.csv', 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        if cnt == 0:
            print(row)
        if cnt > 0:

            labels_list.append(row[0])
            weight_list.append(row[4])
            clasee_names.append(row[7])
        cnt = cnt + 1

experiment = 'cosloss_15_69_cutOut_cosine'
# False for euclidean nearest neighbor, True for cosine similarity distance
is_use_cosine_similarity = True
train_labels_tsv = 'labels/cosloss_15_69_cutOut_train_no_relu.tsv'
valid_labels_tsv = 'labels/cosloss_15_69_cutOut_valid_no_relu.tsv'
train_embedding_csv = 'embeddings/cosloss_15_69_cutOut_train_no_relu.csv'
valid_embedding_csv = 'embeddings/cosloss_15_69_cutOut_valid_no_relu.csv'


train_embeddings = np.genfromtxt(train_embedding_csv, delimiter=',')
valid_embeddings = np.genfromtxt(valid_embedding_csv, delimiter=',')
train_labels = np.genfromtxt(train_labels_tsv, delimiter='\t')
valid_labels = np.genfromtxt(valid_labels_tsv, delimiter='\t')

N_valid = valid_embeddings.shape[0]
N_train = train_embeddings.shape[0]

confusion_mat_data = np.zeros((N_valid,2), dtype=np.int)


lines = 'valid \t train\n'
for n in range(N_valid):
    min_distance = 99999999999
    best_label = -1
    print(n)
    valid_vec = valid_embeddings[n,:]
    for k in range(N_train):
        train_vec = train_embeddings[k,:]
        if is_use_cosine_similarity:
            dist = -1*findCosineSimilarity(valid_vec,train_vec)
        else:
            dist = findEuclideanDistance(valid_vec, train_vec)

        if dist < min_distance:
            min_distance = dist
            #print('dist = {}'.format(dist))
            best_label = train_labels[k]
    confusion_mat_data[n,0] = valid_labels[n]
    confusion_mat_data[n, 1] = best_label
    valid_class = clasee_names[np.int(confusion_mat_data[n,0])].replace(',', '_')
    train_class = clasee_names[np.int(confusion_mat_data[n, 1])].replace(',', '_')
    lines = lines + '{} \t {}\n'.format(valid_class, train_class)

# in the following file classes are represented by numbers
np.savetxt('conf_mat_data/' + experiment + '_data.csv', confusion_mat_data, delimiter=",")

with open('conf_mat_data/' + experiment + '_labels.csv', "w") as text_file:
    text_file.write(lines)

# print accuracy result
vec0 = confusion_mat_data[:,0]
vec1 = confusion_mat_data[:,1]

num_no_zero = np.count_nonzero(vec0-vec1)
equal_elements = vec0.shape[0] - num_no_zero
accuracy = 100*equal_elements/vec0.shape[0]
print('accuracy= {}'.format(accuracy))




