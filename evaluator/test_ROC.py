

import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt

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

def get_fpr_tpr_auc_based_on_embeddings(embedding_file, labels_file):
    # read embeddings
    embeddings = np.genfromtxt(embedding_file, delimiter=',')
    N = embeddings.shape[0]
    temp_scores = np.zeros((N,N),dtype=np.float32)
    temp_labels_scores = np.zeros((N,N),dtype=np.bool)
    labels = np.genfromtxt(labels_file, delimiter='\t')


    # calculate cosine similarity matrix

    for k in range(N):
        print(k)
        for m in range(N):
            #print(m)
            temp_scores[k, m] = findCosineSimilarity(embeddings[k], embeddings[m])
            if labels[k] == labels[m]:
                temp_labels_scores[k,m] = 1

    scores_vec = np.reshape(temp_scores, (N*N))
    labels_vec = np.reshape(temp_labels_scores, (N*N))
    labels_vec = labels_vec.astype(dtype=np.bool)

    fpr, tpr, _ = metrics.roc_curve(labels_vec,  scores_vec)
    auc = metrics.roc_auc_score(labels_vec, scores_vec)
    return fpr,tpr,auc

embedding_files = ['embeddings_cosFace_imagenet.csv','embeddings_cosFace_antique.csv','embeddings_cosFace_imagenet_vgg.csv']
labels1 = ['inception_imagenet','inception_antique','vgg19_imagenet']
labels_files = ['labels_cosFace_imagenet.tsv','labels_cosFace_antique.tsv','labels_cosFace_imagenet_vgg.tsv']
fig = plt.figure()
for k in range(3):
    fpr, tpr, auc = get_fpr_tpr_auc_based_on_embeddings(embedding_files[k],labels_files[k])
    plt.plot(fpr,tpr,label=labels1[k] + ' auc={:.3f}'.format(auc) )
    #plt.legend(loc=4)



plt.legend()
plt.xlabel('false positive rate')
plt.ylabel('true positive rate')
plt.title('roc_inception_vgg_imagenet_antique')
plt.savefig('roc_inception_vgg_imagenet_antique.png')
plt.show()

# plt.plot(fpr,tpr,label="data 1, auc={:.3f}".format(auc))
# plt.legend(loc=4)
# plt.show()

# generate same/no same labels

# plot ROC curve - sklearn function
# for each threshold calc TPR vs FPR
# calc AUC or equal error point