from base.base_trainer import BaseTrain
import os
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping, LearningRateScheduler, ReduceLROnPlateau, LambdaCallback
from trainers.learning_rate_scd import step_decay_wrapper
import numpy as np
from data_loader.default_generator import get_testing_generator
from evaluator.get_valid_nearest_neighbor import eval_model
import json
import csv
from json import encoder
from sklearn.utils import class_weight
import matplotlib.pyplot as plt




class CosLossModelTrainer(BaseTrain):
    def __init__(self, model, train_generator, valid_generator, config):
        super(CosLossModelTrainer, self).__init__(model, train_generator, config)
        self.valid_generator = valid_generator
        self.callbacks = []
        self.loss = []
        self.val_loss = []
        self.step_decay_function = step_decay_wrapper(self.config)
        encoder.FLOAT_REPR = lambda o: format(o, '.2f')

        cnt = 0
        self.period_dict = {}
        self.site_dict = {}
        with open(self.config.data_loader.classes_info_csv_file, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                if cnt > 0:
                    self.period_dict[int(row[0])] = int(row[5])
                    self.site_dict[int(row[0])] = int(row[6])
                cnt = cnt + 1


        self.init_callbacks()


    def init_callbacks(self):
        #filepath = os.path.join(self.config.callbacks.checkpoint_dir,
        #                        '%s-{epoch:02d}-{val_loss:.2f}.hdf5' % self.config.exp.name)
        if self.config.callbacks.is_save_model:
            self.callbacks.append(
                ModelCheckpoint(
                    #self.config.callbacks.checkpoint_dir, '%s-{epoch:02d}-{val_loss:.2f}.hdf5' % self.config.exp.name
                    filepath=os.path.join(self.config.callbacks.checkpoint_dir, '%s-{epoch:02d}-{val_embeddings_loss:.2f}.hdf5' % self.config.exp.name),
                    monitor=self.config.callbacks.checkpoint_monitor,
                    mode=self.config.callbacks.checkpoint_mode,
                    save_best_only=self.config.callbacks.checkpoint_save_best_only,
                    save_weights_only=self.config.callbacks.checkpoint_save_weights_only,
                    verbose=self.config.callbacks.checkpoint_verbose,
                )
            )

        self.callbacks.append(
            TensorBoard(
                log_dir=self.config.callbacks.tensorboard_log_dir,
                write_graph=self.config.callbacks.tensorboard_write_graph,
            )
        )

        if self.config.trainer.is_early_stop:
            self.callbacks.append(
                EarlyStopping(
                    monitor='val_loss', min_delta=0.05, patience=9, verbose=1, mode='auto')
            )

        if self.config.trainer.is_change_lr:
            if self.config.trainer.learning_rate_schedule_type == 'ReduceLROnPlateau':
                self.callbacks.append(
                    ReduceLROnPlateau(monitor='val_loss', factor=self.config.trainer.lr_decrease_factor,
                                                patience=2, min_lr=1e-12)
                )
            elif self.config.trainer.learning_rate_schedule_type == 'LearningRateScheduler':
                self.callbacks.append(
                    LearningRateScheduler(self.step_decay_function)
            )


        if self.config.model.loss == 'triplet':
            if self.config.model.batch_type == 'hard':
                self.callbacks.append(
                    LambdaCallback(on_epoch_end=lambda epoch, logs: self.custom_epoch_end(epoch,logs,'triplet_hard'),
                        on_train_end=lambda logs: self.json_log.close())
                    )
            else:
                self.callbacks.append(
                    LambdaCallback(on_epoch_end=lambda epoch, logs: self.custom_epoch_end(epoch,logs,'triplet_all'),
                        on_train_end=lambda logs: self.json_log.close())
                    )
        elif self.config.model.loss == 'cosface':
            self.callbacks.append(
                LambdaCallback(on_epoch_end=lambda epoch, logs: self.custom_epoch_end(epoch,logs, 'cosface'),
                               on_train_end=lambda logs: self.json_log.close())
            )
        elif self.config.model.loss == 'softmax':
            self.callbacks.append(
                LambdaCallback(on_epoch_end=lambda epoch, logs: self.custom_epoch_end(epoch,logs, 'softmax'),
                               on_train_end=lambda logs: self.json_log.close())
            )


        #if hasattr(self.config,"comet_api_key"):
        #    from comet_ml import Experiment
        #    experiment = Experiment(api_key=self.config.comet_api_key, project_name=self.config.exp_name)
        #    experiment.disable_mp()
        #    experiment.log_multiple_params(self.config)
        #    self.callbacks.append(experiment.get_keras_callback())

    def train(self):
        self.json_log = open(self.config.callbacks.tensorboard_log_dir + '/loss_log.json', mode='wt', buffering=1)
        #sorted_labels = sorted(self.generator.total_labels)
        #class_weights = class_weight.compute_class_weight('balanced', np.unique(sorted_labels), sorted_labels)
        #class_weights_list = []
        #class_weights_list.append(class_weights)
        #class_weights_list.append(class_weights)
        #print(class_weights_list)


        history = self.model.fit_generator(
            self.generator,
            epochs=self.config.trainer.num_epochs,
            steps_per_epoch=len(self.generator),
            validation_data=self.valid_generator,
            validation_steps=len(self.valid_generator),
            use_multiprocessing = True,
            max_queue_size=10,
            workers=5,
            callbacks=self.callbacks,
            #class_weight=class_weights_list,
            verbose=1)

        self.loss.extend(history.history['loss'])
        self.val_loss.extend(history.history['val_loss'])



    def get_accuracy(self,epoch, isSaveEmbeddings = False ):
        # get accuracy, using default generators
        self.config['data_loader']['data_dir_train'] = self.config['data_loader']['data_dir_train_test']
        self.config['data_loader']['data_dir_valid'] = self.config['data_loader']['data_dir_valid_test']
        train_generator = get_testing_generator(self.config, True)
        valid_generator = get_testing_generator(self.config, False)
        generators = [train_generator, valid_generator]
        generators_id = ['_train', '_valid']

        for m, generator in enumerate(generators):

            # print(generators_id[m])
            # for filename in generator.filenames:
            #     print(filename)

            batch_size = self.config['data_loader']['batch_size']
            num_of_images = len(generator) * (batch_size)
            labels = np.zeros((num_of_images, 1), dtype=np.int)
            labels_period = np.zeros((num_of_images, 1), dtype=np.int)
            labels_site = np.zeros((num_of_images, 1), dtype=np.int)
            predication = np.zeros((num_of_images, int(self.config.model.embedding_dim)), dtype=np.float32)

            label_map = (generator.class_indices)
            label_map = dict((v, k) for k, v in label_map.items())  # flip k,v

            cur_ind = 0
            for k in range(len(generator)):
                if (k % 10) == 0:
                    print(k)
                x, y_true_ = generator.__getitem__(k)
                y_true = [label_map[x] for x in y_true_]
                y_pred = self.model.predict(x)

                if self.config.model.num_of_outputs > 1:
                    y_pred = y_pred[1]

                num_of_items = y_pred.shape[0]
                predication[cur_ind: cur_ind + num_of_items, :] = y_pred
                labels[cur_ind: cur_ind + num_of_items, :] = np.expand_dims(y_true, axis=1)
                cur_ind = cur_ind + num_of_items


            predication = predication[:cur_ind, :]
            labels = labels[:cur_ind, :]

            for kk in range(labels.shape[0]):
                labels_period[kk,0] = self.period_dict[labels[kk,0]]
                labels_site[kk, 0] = self.site_dict[labels[kk, 0]]

            labels_period = labels_period[:cur_ind, :]
            labels_site = labels_site[:cur_ind, :]

            if m == 0:
                train_labels = labels
                train_labels_period = labels_period
                train_labels_site = labels_site
                train_prediction = predication
            else:
                valid_labels = labels
                valid_labels_period = labels_period
                valid_labels_site = labels_site
                valid_prediction = predication

            if isSaveEmbeddings:
                np.savetxt('evaluator/labels/' + self.config.exp.name + generators_id[m] + str(epoch) +  '.tsv', labels, delimiter=',')
                np.savetxt('evaluator/embeddings/' + self.config.exp.name + generators_id[m] + str(epoch) + '.csv', predication,
                       delimiter=',')

        accuracy = eval_model(train_prediction, valid_prediction, train_labels, valid_labels, self.config.exp.name,
                              is_save_files=False)
        print('accuracy_class = {0:.3f}'.format(accuracy))


        accuracy_period = eval_model(train_prediction, valid_prediction, train_labels_period, valid_labels_period, self.config.exp.name,
                              is_save_files=False)
        print('accuracy_periods = {0:.3f}'.format(accuracy_period))


        accuracy_site = eval_model(train_prediction, valid_prediction, train_labels_site, valid_labels_site, self.config.exp.name,
                              is_save_files=False)
        print('accuracy_sites = {0:.3f}'.format(accuracy_site))

        return accuracy, accuracy_period, accuracy_site


    def custom_epoch_end(self,epoch,logs,type):
        #print(np.dtype(logs['lr']))
        #print(np.dtype(logs['val_loss']))
        acc, acc_period, acc_site = self.get_accuracy(epoch)
        #acc = 0
        #acc_period = 0
        #acc_site = 0
        if type == 'cosface':
            self.json_log.write(
                json.dumps(
                    {'epoch': epoch, 'loss': logs['embeddings_loss'], 'val_loss': logs['val_loss'],
                     'acc': acc, 'acc_period': acc_period , 'acc_site': acc_site, 'lr': logs['lr'].astype('float64')}) + '\n')
        elif type == 'softmax':
            self.json_log.write(
                json.dumps(
                    {'epoch': epoch, 'loss': logs['out_loss'], 'val_loss': logs['val_loss'],
                     'acc_out': logs['out_acc'],'acc_out_val': logs['val_out_acc'],
                     'acc': acc, 'acc_period': acc_period , 'acc_site': acc_site, 'lr': logs['lr'].astype('float64')}) + '\n')
        elif type == 'triplet_all':
            self.json_log.write(
                json.dumps(
                    {'epoch': epoch, 'loss': logs['loss'], 'val_loss': logs['val_loss'], 'acc': acc, 'acc_period': acc_period,
                     'acc_site': acc_site, 'positive_fraction': logs['embeddings_positive_fraction'],
                     'val_positive_fraction': logs['val_embeddings_positive_fraction']}) + '\n')
        elif type == 'triplet_hard':
            self.json_log.write(
                json.dumps(
                    {'epoch': epoch, 'loss': logs['embeddings_loss'], 'val_loss': logs['val_embeddings_loss'], 'acc': acc,'acc_period': acc_period,
                     'acc_site': acc_site, 'hard_pos_dist': logs['embeddings_hardest_pos_dist'], 'hard_neg_dist': logs['embeddings_hardest_neg_dist']}) + '\n'),
