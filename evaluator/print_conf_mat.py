import numpy as np
from sklearn.metrics import confusion_matrix
import csv
import matplotlib.pyplot as plt
import itertools

def plot_2_conf_mat(cm1,cm2):
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
    ax1.imshow(cm1)
    #ax1.imshow(cm1)
    ax1.set_ylabel('True label', fontsize=18)
    ax1.set_xlabel('Predicted label', fontsize=18)
    ax1.set_title('sorted by number of images', fontsize=18)
    im = ax2.imshow(cm2)
    #im = ax2.imshow(cm2)
    ax2.set_xlabel('Predicted label', fontsize=18)
    ax2.set_title('sorted by periods', fontsize=18)
    ax1.tick_params(labelsize=14)
    ax2.tick_params(labelsize=14)


    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.2, 0.02, 0.6])
    fig.colorbar(im, cax=cbar_ax)
    #plt.tight_layout()

    plt.show()

def plot_confusion_matrix(cm, classes,experient,
                          normalize=False,
                          title='Confusion matrix, labels are sites',
                          cmap=plt.cm.Blues,):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    #plt.title(title)
    plt.colorbar()

    plt.ylabel('True label', fontsize=14)
    plt.xlabel('Predicted label', fontsize=14)
    plt.tight_layout()
    plt.savefig('results/' + experient + '.png')
    return cm

def get_site_period_str(class_name,class_mode):
    if class_mode == 'site_period':
        period, site = class_name.split('_')
    elif class_mode == 'site_period_sorted':
        period, site = class_name.split('_')
    elif class_mode == 'period_sorted':
        period = class_name
        site = ''
    elif class_mode == 'site':
        site = class_name
        period = ''
    else:
        print('invalid class mode')
        raise

    return period,site

def get_confusion_matrix(experiment,data,class_mode,classes_csv_file = '', num_of_relevant_neighbors = -1):
    cnt = 0
    class_names = {}
    if classes_csv_file == '':
        classes_csv_file = '../data_loader/classes_top200.csv'


    with open(classes_csv_file, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if cnt > 0:
                if class_mode == 'site_period':
                    class_names[int(row[0])] = row[1]
                elif class_mode == 'site_period_sorted':
                    class_names[int(row[8])] = row[1]
                elif class_mode == 'period_sorted':
                    class_names[int(row[5])] = row[3]
                elif class_mode == 'site':
                    class_names[int(row[6])] = row[4]
                else:
                    print('invalid class mode')
                    raise
            cnt = cnt + 1
    if num_of_relevant_neighbors == -1:
        y_true = data[:,0]
        y_pred = data[:,1]
    else:
        N = data[:,0].shape[0]
        y_true = np.zeros((N*num_of_relevant_neighbors, ), dtype=np.int32)
        y_pred = np.zeros((N * num_of_relevant_neighbors,), dtype=np.int32)
        for mm in range(num_of_relevant_neighbors):
            y_true[mm*N:(mm+1)*N] = data[:,0]
            y_pred[mm*N:(mm + 1) * N] = data[:, mm + 1]
        print('aaa')
        print(y_true.shape)
    conf = confusion_matrix(y_true, y_pred)

    if class_mode == 'site_period':
        class_names[int(row[0])] = row[1]
        conf_norm = plot_confusion_matrix(conf, [], experiment, title='label = ' + class_mode,
                                          normalize=True)
    elif class_mode == 'site_period_sorted':
        class_names[int(row[8])] = row[1]
        conf_norm = plot_confusion_matrix(conf, [], experiment, title='label = ' + 'site_period, sorted by period',
                                          normalize=True)
    elif class_mode == 'period_sorted':
        class_names[int(row[5])] = row[3]
        conf_norm = plot_confusion_matrix(conf, [], experiment, title='label = ' + 'period, sorted',
                                          normalize=True)
    elif class_mode == 'site':
        class_names[int(row[6])] = row[4]
        conf_norm = plot_confusion_matrix(conf, [], experiment, title='label = ' + class_mode,
                                        normalize=True)
    else:
        print('invalid class mode')
        raise



    plt.show()

    N = 10
    lines = ''
    for k in range(conf_norm.shape[0]):
        vec = conf_norm[k,:]
        ind_max = np.flip(np.argsort(vec), axis=0)
        ind_max = ind_max[:N]

        scores = vec[ind_max]
        scores = np.trim_zeros(scores, 'b')

        period, site = get_site_period_str(class_names[k], class_mode)

        lines = lines + 'true_id, {}, true_site, {}, true_period, {}\n'.format(k, site, period)

        for m in range(scores.shape[0]):
            pred_id = ind_max[m]
            period, site = get_site_period_str(class_names[pred_id], class_mode)
            temp_str = 'pred_id, {}, pred_site, {}, pred_period, {}, pred_score, {:0.3f}\n'.format(pred_id, site, period, scores[m])
            lines = lines + temp_str

        lines = lines + '\n'

    with open('results/summary_' + experient + '.csv', "w") as text_file:
        text_file.write(lines)

    return conf_norm




experient = 'sie_period_sorted_efficientNetB3_softmax_concat_embeddings_10_rp1500'
data = np.genfromtxt('conf_mat_data/efficientNetB3_conc10_rp_full_neighbor_site_period__data.csv', delimiter=',')
data_period_sorted = np.zeros_like(data, dtype=np.int32)
cnt = 0
class_dict = {}
with open('../data_loader/classes_top200.csv', 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        if cnt > 0:
            class_dict[int(row[0])] = int(row[8])
        cnt = cnt + 1
for i in range(data.shape[0]):
    for j in range(data.shape[1]):
        data_period_sorted[i,j] = class_dict[data[i,j]]


cm_site_period_sorted_1 = get_confusion_matrix(experient,data_period_sorted,'site_period_sorted')
cm_site_period_sorted = get_confusion_matrix(experient,data_period_sorted,'site_period_sorted',num_of_relevant_neighbors=30)


experient = 'site_period_efficientNetB3_softmax_concat_embeddings_10_rp1500'
data = np.genfromtxt('conf_mat_data/efficientNetB3_conc10_rp_full_neighbor_site_period__data.csv', delimiter=',')
cm_site_period = get_confusion_matrix(experient,data,'site_period')
plot_2_conf_mat(cm_site_period,cm_site_period_sorted_1)
assert 0 == 1

experient = 'period_efficientNetB3_softmax_concat_embeddings_10_rp1500'
data = np.genfromtxt('conf_mat_data/efficientNetB3_softmax_concat_embeddings_10_rp1500_period_data.csv', delimiter=',')
cm_period_sorted = get_confusion_matrix(experient,data,'period_sorted')

plot_2_conf_mat(cm_site_period,cm_site_period_sorted)

experient = 'site_efficientNetB3_softmax_concat_embeddings_10_rp1500'
data = np.genfromtxt('conf_mat_data/efficientNetB3_softmax_concat_embeddings_10_rp1500_site_data.csv', delimiter=',')
cm_site = get_confusion_matrix(experient,data,'site')

fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4)
ax1.imshow(cm_site_period, interpolation='nearest',cmap=plt.cm.Blues)
ax1.set_title('label = site_period')
ax1.set(xlabel='Predicted label\n\n(a)', ylabel='True label')
ax2.imshow(cm_site_period_sorted, interpolation='nearest',cmap=plt.cm.Blues)
#np.savetxt('confusion_ste_period_sorted.csv', cm_site_period_sorted, delimiter=",")
ax2.set_title('label = site_period, sorted by period')
ax2.set(xlabel='Predicted label\n\n(b)', ylabel='True label')
ax3.imshow(cm_period_sorted, interpolation='nearest',cmap=plt.cm.Blues)
ax3.set_title('label = period, sorted')
ax3.set(xlabel='Predicted label\n\n(c)', ylabel='True label')
ax4.imshow(cm_site, interpolation='nearest',cmap=plt.cm.Blues)
ax4.set_title('label = site')
ax4.set(xlabel='Predicted label\n\n(d)', ylabel='True label')
plt.show()
