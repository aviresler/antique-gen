import numpy as np
from sklearn.metrics import confusion_matrix
import csv
import matplotlib.pyplot as plt
import itertools


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
    plt.title(title)
    plt.colorbar()

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
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

def get_confusion_matrix(experiment,data,class_mode,classes_csv_file = ''):
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

    y_true = data[:,0]
    y_pred = data[:,1]

    #classes = list(class_names.keys())
    #classes = np.sort(classes)

    conf = confusion_matrix(y_true, y_pred)
    conf_norm = plot_confusion_matrix(conf,[],experiment,title='Confusion matrix, label mode is ' + class_mode ,normalize=True)

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




experient = 'site_period_sorted_14_5_57.8'
data = np.genfromtxt('conf_mat_data/triplet_14_5_57.8_site_period_data.csv', delimiter=',')
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


get_confusion_matrix(experient,data_period_sorted,'site_period_sorted')

experient = 'site_period_14_5_57.8'
data = np.genfromtxt('conf_mat_data/triplet_14_5_57.8_site_period_data.csv', delimiter=',')
get_confusion_matrix(experient,data,'site_period')

experient = 'period_14_5_57.8'
data = np.genfromtxt('conf_mat_data/triplet_14_5_57.8_period_data.csv', delimiter=',')
get_confusion_matrix(experient,data,'period_sorted')

experient = 'site_14_5_57.8'
data = np.genfromtxt('conf_mat_data/triplet_14_5_57.8_site_data.csv', delimiter=',')
get_confusion_matrix(experient,data,'site')