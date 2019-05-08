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

    print(cm)
    #np.savetxt('confusion_mat.txt', cm, delimiter=',')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    print('here')
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    # plt.xticks(tick_marks, classes, rotation=45)
    # plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    # for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    #     plt.text(j, i, format(cm[i, j], fmt),
    #              horizontalalignment="center",
    #              color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig('results/' + experient + '.png')
    return cm

experient = 'final_site'
data = np.genfromtxt('conf_mat_data/final_sites_data.csv', delimiter=',')
cnt = 0
period_list = []
labels_list = []
class_names = {}
with open('../data_loader/classes_top200.csv', 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        if cnt == 0:
            print(row)
        if cnt > 0:
            class_names[int(row[6])] = row[4]
            #period_list.append(row[3])

            #labels_list.append(row[0])
            #class_names.append()
        cnt = cnt + 1

# period_set = set(period_list)
# for prd in period_set:
#     print(prd)

y_true = data[:,0]
y_pred = data[:,1]

conf = confusion_matrix(y_true, y_pred)
conf_norm = plot_confusion_matrix(conf,labels_list,experient,normalize=True)

N = 10
lines = ''
for k in range(120):
    vec = conf_norm[k,:]
    ind_max = np.flip(np.argsort(vec), axis=0)
    ind_max = ind_max[:N]

    scores = vec[ind_max]
    scores = np.trim_zeros(scores, 'b')

    #period, site  = class_names[k].split('_')
    period = ' '
    #site = site.replace(',','_')
    site = class_names[k]
    #period = period.replace(',', '_')
    lines = lines + 'true_id, {}, true_site, {}, true_period, {}\n'.format(k, site, period)

    for m in range(scores.shape[0]):
        pred_id = ind_max[m]
        #period, site = class_names[pred_id].split('_')
        #period = class_names[pred_id]
        period = ' '
        #site = site.replace(',', '_')
        site = class_names[pred_id]
        #period = period.replace(',', '_')
        temp_str = 'pred_id, {}, pred_site, {}, pred_period, {}, pred_score, {:0.3f}\n'.format(pred_id, site, period, scores[m])
        lines = lines + temp_str

    lines = lines + '\n'



with open('results/summary_' + experient + '.csv', "w") as text_file:
    text_file.write(lines)


# fig = plt.figure()
# ax = fig.add_subplot(111)
# cax = ax.matshow(conf)
# fig.colorbar(cax)

# ticks = np.arange(0, 4, 1)
# ax.set_xticks(ticks)
# ax.set_yticks(ticks)
# ax.set_xticklabels(classes)
# ax.set_yticklabels(classes)
plt.show()
# fig.savefig('similarity_overfit.png')
# np.savetxt('similarity_overfit.csv', results, delimiter=',')
# np.set_printoptions(precision=3, suppress=True)

#print(a.shape)
