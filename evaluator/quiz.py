import os
import numpy as np
import pandas as pd
import random
from get_valid_nearest_neighbor import eval_model_from_csv_files, eval_model_topk, eval_model, eval_model_per_period_group, eval_model_per_period_group_quiz_check
import csv
import matplotlib.pyplot as plt
import shutil


def pick_random_sample(num_of_sampled_images,period_group_col ):
    # get number of groups from classes csv
    df = pd.read_csv('../data_loader/classes_top200.csv')
    period_groups = df[period_group_col]
    uniqe_period_groups = np.unique(period_groups.values)
    images_per_group = int(np.ceil(num_of_sampled_images/uniqe_period_groups.size))

    sampled_files = []
    for group in uniqe_period_groups:
        # generate list of images files in each group
        group_folders = df[period_groups == group]
        classes = group_folders['id'].values
        files = []
        for cls in classes:
            cls_files = os.listdir( '../data_loader/data/data_16_6/site_period_top_200_bg_removed/valid/' + str(cls) + '/' )
            cls_files = [(lambda x: str(cls) + '/' + x)(x) for x in cls_files]
            files.append(cls_files)
        files = [item for sublist in files for item in sublist]

        # get artifacts list by getting only the main view (file name ends with _0.jpg)
        artifacts_files = list(filter(lambda x: x.endswith('_0.jpg'), files))
        temp = random.sample(artifacts_files, images_per_group)

        sampled_files.append(temp)

    sampled_files = [item for sublist in sampled_files for item in sublist]
    random.shuffle(sampled_files)
    random.shuffle(sampled_files)

    if len(sampled_files) > num_of_sampled_images:
        sampled_files = random.sample(sampled_files, num_of_sampled_images)

    return sampled_files

def predict(sample, train_embeddings, valid_embeddings, train_labels, valid_labels, period_group_col, is_print = True):

    experiment = 'tets'
    # get relevant embeddings according to sample
    valid_files = []
    with open('valid_file_names.csv', 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            valid_files.append(row[0])

    ind = []
    for file in sample:
        ind.append(valid_files.index(file))
    #valid_embeddings = valid_embeddings[ind, :]
    #valid_labels = valid_labels[ind]

    classes_csv_file = '../data_loader/classes_top200.csv'

    train_labels_period = np.zeros_like(train_labels, dtype=np.int32)
    valid_labels_period = np.zeros_like(valid_labels, dtype=np.int32)
    train_labels_site = np.zeros_like(train_labels, dtype=np.int32)
    valid_labels_site = np.zeros_like(valid_labels, dtype=np.int32)
    train_labels_fine_period = np.zeros_like(train_labels, dtype=np.int32)
    valid_labels_fine_period = np.zeros_like(valid_labels, dtype=np.int32)

    cnt = 0
    period_dict = {}
    group_period_name_dict = {}
    if period_group_col == 'period_group_rough':
        col = 9
    else:
        col = 10

    site_dict = {}
    period_name_dict = {}
    clasess_dict = {}
    with open(classes_csv_file, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if cnt > 0:
                site_dict[int(row[0])] = int(row[6])
                period_dict[int(row[0])] = int(row[5])
                clasess_dict[int(row[0])] = int(row[10])
                if not (int(row[col]) in  group_period_name_dict):
                    group_period_name_dict[int(row[col])] = row[3]
                if not (int(row[5]) in  period_name_dict):
                    period_name_dict[int(row[5])] = row[3]
            cnt = cnt + 1
    group_period_name_list = [v for v in group_period_name_dict.values()]

    for i, val_label in enumerate(valid_labels):
        valid_labels_period[i] = period_dict[val_label]
        valid_labels_site[i] = site_dict[val_label]
        valid_labels_fine_period[i] = clasess_dict[val_label]

    for i, tr_label in enumerate(train_labels):
        train_labels_period[i] = period_dict[tr_label]
        train_labels_site[i] = site_dict[tr_label]
        train_labels_fine_period[i] = clasess_dict[tr_label]

    #plt.figure(figsize=(14, 10))
    accuracy, accuracy_per_period, true_label, pred_label = eval_model_per_period_group(train_embeddings, valid_embeddings, train_labels_fine_period, valid_labels_fine_period,period_group_col)


    if is_print:
        y_pos = np.arange(len(accuracy_per_period))
        plt.bar(y_pos, accuracy_per_period, align='center', alpha=0.5)
        plt.axhline(y=accuracy)
        plt.legend(['test accuracy','group accuracy'],fontsize = 12)
        plt.xticks(y_pos, group_period_name_list, rotation=90,fontsize = 14)
        plt.tight_layout()

        plt.ylabel('accuracy [%]',fontsize = 18)
        plt.title('fine period group accuracy, test accuracy = ' + str(round(accuracy, 2)) + ' [%]'.format(accuracy),fontsize = 18)
        plt.savefig('accuracy_per_period_complete_valid_set.jpg')

        plt.show()



    return accuracy

def check_quiz_results(answers_csv, predictions_csv):
    clasess_dict = {}
    cnt = 0
    with open('../data_loader/classes_top200.csv', 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if cnt > 0:
                clasess_dict[row[3]] = int(row[10])
            cnt = cnt + 1

    cnt = 0
    answers = []
    with open(answers_csv, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            answers.append(clasess_dict[row[0]])
            cnt = cnt + 1

    cnt = 0
    predictions = []
    with open(predictions_csv, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            predictions.append(clasess_dict[row[0]])
            cnt = cnt + 1

    cnt = 0
    period_dict = {}
    group_period_name_dict = {}
    col = 10
    site_dict = {}
    period_name_dict = {}
    with open('../data_loader/classes_top200.csv', 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if cnt > 0:
                site_dict[int(row[0])] = int(row[6])
                period_dict[int(row[0])] = int(row[5])
                if not (int(row[col]) in  group_period_name_dict):
                    group_period_name_dict[int(row[col])] = row[3]
                if not (int(row[5]) in  period_name_dict):
                    period_name_dict[int(row[5])] = row[3]
            cnt = cnt + 1
    group_period_name_list = [v for v in group_period_name_dict.values()]

    #print(answers)
    #print(predictions)



    accuracy, accuracy_per_period, true_label, pred_label = eval_model_per_period_group_quiz_check(np.asarray(answers),np.asarray(predictions), 'period_group_fine' )
    print(str(round(accuracy,2)))


    y_pos = np.arange(len(accuracy_per_period))
    plt.bar(y_pos, accuracy_per_period, align='center', alpha=0.5)
    plt.axhline(y=accuracy)
    #plt.axhline(y=69.84,color='r')
    plt.legend(['test accuracy','group accuracy'],fontsize = 12)
    plt.xticks(y_pos, group_period_name_list, rotation=90,fontsize = 14)
    plt.tight_layout()

    plt.ylabel('accuracy [%]',fontsize = 18)
    plt.title('Quiz results: fine period group accuracy, average accuracy = ' + str(round(accuracy,2)) + ' [%]'.format(accuracy),fontsize = 18)
    plt.savefig('accuracy_per_period_complete_valid_set.jpg')

    plt.show()




def run_many_random_samples():
    train_embeddings_csv = 'embeddings/efficientNetB3_softmax_concat_embeddings_10_rp1500_train.csv'
    valid_embeddings_csv = 'embeddings/efficientNetB3_softmax_concat_embeddings_10_rp1500_valid.csv'
    train_labels_tsv = 'labels/efficientNetB3_softmax_averaged_embeddings_train.tsv'
    valid_labels_tsv = 'labels/efficientNetB3_softmax_averaged_embeddings_valid.tsv'

    train_embeddings = np.genfromtxt(train_embeddings_csv, delimiter=',')
    valid_embeddings = np.genfromtxt(valid_embeddings_csv, delimiter=',')
    train_labels = np.genfromtxt(train_labels_tsv, delimiter='\t')
    valid_labels = np.genfromtxt(valid_labels_tsv, delimiter='\t')

    test_results = []
    min_dist_from_mean = 999
    for k in range(500):
        sample = pick_random_sample(52, 'period_group_rough')
        period_acc = predict(sample, train_embeddings, valid_embeddings, train_labels, valid_labels,'period_group_rough', is_print=False)
        dist_from_mean = np.abs(period_acc - 71.9)
        if  dist_from_mean < min_dist_from_mean:
            test = sample
            min_dist_from_mean = dist_from_mean
            print('update test')
            print(period_acc)

        test_results.append(period_acc)
        print(period_acc)

    test_results_np = np.array(test_results)

    _ = plt.hist(test_results_np, bins=15)
    plt.title("Histogram of rough samples accuracy, sample_size = 52, mean accuracy = 71.9")
    print('mean')
    print(np.mean(test_results_np))
    plt.savefig('sample_52_accuracy_hist.png')
    plt.show()

    with open('test_file_names_52.csv', "w") as outfile:
        for entries in test:
            outfile.write(entries)
            outfile.write("\n")

    print('test period accuracy')
    period_acc = predict(test, train_embeddings, valid_embeddings, train_labels, valid_labels,'period_group_rough', is_print=False)
    print(period_acc)


def get_test_results(period_group):
    train_embeddings_csv = 'embeddings/efficientNetB3_softmax_concat_embeddings_10_rp1500_train.csv'
    valid_embeddings_csv = 'embeddings/efficientNetB3_softmax_concat_embeddings_10_rp1500_valid.csv'
    train_labels_tsv = 'labels/efficientNetB3_softmax_averaged_embeddings_train.tsv'
    valid_labels_tsv = 'labels/efficientNetB3_softmax_averaged_embeddings_valid.tsv'

    train_embeddings = np.genfromtxt(train_embeddings_csv, delimiter=',')
    valid_embeddings = np.genfromtxt(valid_embeddings_csv, delimiter=',')
    train_labels = np.genfromtxt(train_labels_tsv, delimiter='\t')
    valid_labels = np.genfromtxt(valid_labels_tsv, delimiter='\t')

    sample = get_test_files_list(period_group)
    print(sample)

    period_acc = predict(sample, train_embeddings, valid_embeddings, train_labels, valid_labels,period_group, is_print=True)

    print(period_acc)


def get_test_files_list(period_group):
    sample = []
    if period_group == 'period_group_rough':
        filename = 'test_file_names_52.csv'
    else:
        filename = 'test_file_names_63.csv'

    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            sample.append(row[1])
    return sample


def generate_test_images(period_group):
    files = get_test_files_list(period_group)
    for m, file in enumerate(files):
        shutil.copy('../data_loader/data/data_16_6/site_period_top_200_bg_removed/valid/' + file, 'quiz/rough/test_images/' + str(m) + '.jpg' )
        print(file)
    # copy files and change names
    print(files)
    #for file in files:

def plot_statistics():
    data = np.genfromtxt('quiz/fine/check_hist.csv', delimiter=',', skip_header=True)
    #bins = np.linspace(0, 3, 4)
    plt.hist(data[:,0], [-0.5, 1.5, 2.5, 3.5], alpha=0.5, label='x')
    print(np.where(data[:,4] == 3)[0].shape)
    #pyplot.hist(y, bins, alpha=0.5, label='y')
    #pyplot.legend(loc='upper right')
    #plt.show()
    #print(data.shape)


if __name__ == '__main__':
    #run_many_random_samples()
    #files = pick_random_sample(63)
    #print(files)
    #get_test_results('period_group_fine')
    #plot_statistics()
    check_quiz_results('quiz/fine/check_answers.csv', 'quiz/fine/check_model.csv')

    #run_many_random_samples()
    #files = pick_random_sample(52)
    #print(files)
    #print(len(files))
    #generate_test_images('period_group_rough')
    # df = pd.read_csv('../data_loader/classes_top200.csv')
    # period_groups = df['period_group_fine']
    # uniqe_groups = np.unique(period_groups.values)
    # for group in uniqe_groups:
    #     print(group)
    #     df_slice = df[period_groups == group]
    #     periods = df_slice['period'].values
    #     period_list = list(set(periods))
    #     for prd in period_list:
    #         print(prd)

    # period_name_dict = {}
    # cnt = 0
    # with open('../data_loader/classes_top200.csv', 'r') as f:
    #     reader = csv.reader(f)
    #     for row in reader:
    #         if cnt > 0:
    #             if not (int(row[5]) in  period_name_dict):
    #                 period_name_dict[int(row[5])] = row[3]
    #                 print(row[3])
    #         cnt = cnt + 1


