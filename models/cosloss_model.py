from base.base_model import BaseModel
from keras.layers import Dense, Dropout, Conv2D, Activation, MaxPooling2D, Flatten
from keras.models import Sequential
from keras import applications
from keras.models import Model
from keras import regularizers
from keras import optimizers
from keras import backend as K
import models.cos_face_loss
import models.triplet_loss
import numpy as np

class CosLosModel(BaseModel):
    def __init__(self, config):
        super(CosLosModel, self).__init__(config)

        self.build_model()

    def build_model(self):

        if not self.config.model.is_use_dummy_model:

            self.model = applications.inception_resnet_v2.InceptionResNetV2(include_top=True, weights='imagenet',
                                                                       input_shape=(self.config.model.img_width, self.config.model.img_height, 3))

            self.model.layers.pop()
            x = self.model.layers[-1].output

            if self.config.model.is_use_relu_on_embeddings:
                x = Dense(int(self.config.model.embedding_dim), activation="relu", kernel_regularizer=regularizers.l2(self.config.model.final_layer_regularization))(x)
            else:
                x = Dense(int(self.config.model.embedding_dim), kernel_regularizer=regularizers.l2(self.config.model.final_layer_regularization))(x)

            x = Dropout(0.5)(x)
            self.model = Model(input=self.model.input, output=x)

        else:
            input_shape = (299, 299, 3)
            self.model = Sequential()
            self.model.add(Conv2D(32, (3, 3), input_shape=input_shape))
            self.model.add(Activation('relu'))
            self.model.add(MaxPooling2D(pool_size=(2, 2)))

            self.model.add(Conv2D(32, (3, 3)))
            self.model.add(Activation('relu'))
            self.model.add(MaxPooling2D(pool_size=(2, 2)))

            self.model.add(Conv2D(64, (3, 3)))
            self.model.add(Activation('relu'))
            self.model.add(MaxPooling2D(pool_size=(2, 2)))

            self.model.add(Flatten())
            self.model.add(Dense(int(self.config.model.embedding_dim), activation="relu"))

            self.model = Model(input=self.model.input, output=self.model.output)

        if self.config.trainer.learning_rate_schedule_type == 'LearningRateScheduler':
            adam1 = optimizers.Adam(lr=0.0)
        elif self.config.trainer.learning_rate_schedule_type == 'ReduceLROnPlateau':
            print('decrease_platue')
            adam1 = optimizers.Adam(lr=self.config.trainer.learning_rate)
        else:
            print('invalid learning rate configuration')
            raise

        metrics = []
        if self.config.model.loss == 'triplet':
            loss_func = self.triplet_loss_wrapper(self.config.model.margin, self.config.model.is_squared)
            if self.config.model.batch_type == 'hard':
                hardest_pos_dist = self.triplet_loss_wrapper_batch_hard_hardest_pos_dist(self.config.model.margin, self.config.model.is_squared)
                metrics.append(hardest_pos_dist)
                hardest_neg_dist = self.triplet_loss_wrapper_batch_hard_hardest_neg_dist(self.config.model.margin, self.config.model.is_squared)
                metrics.append(hardest_neg_dist)
            if self.config.model.batch_type == 'all':
                pos_fraction = self.triplet_loss_wrapper_batch_all_positive_fraction(self.config.model.margin,
                                                                                         self.config.model.is_squared)
                metrics.append(pos_fraction)
        elif self.config.model.loss == 'cosface':
            if self.config.model.is_use_prior_weights:
                loss_func = self.weighted_coss_loss_wrapper(self.config.model.alpha, self.config.model.scale)
            else:
                loss_func = self.coss_loss_wrapper(self.config.model.alpha, self.config.model.scale)
        else:
            print('invalid loss type')
            raise


        self.model.compile(
              loss=loss_func,
              optimizer=adam1,
              metrics= metrics
        )

    def coss_loss_wrapper(self, alpha, scale):
        def coss_loss1(y_true, y_pred):
            y_true_casted = K.cast(y_true, dtype='int32')
            y_true_cls = y_true_casted[:, 0]
            y_true_period = y_true_casted[:, 1]
            y_true_site = y_true_casted[:, 2]
            cls_loss = models.cos_face_loss.cos_loss(y_pred, y_true_cls, self.config.data_loader.num_of_classes,
                                                     alpha=alpha, scale=scale, reuse=False, name='cls')
            period_loss = models.cos_face_loss.cos_loss(y_pred, y_true_period, self.config.data_loader.num_of_periods,
                                                     alpha=alpha, scale=scale, reuse=False, name='periods')
            site_loss = models.cos_face_loss.cos_loss(y_pred, y_true_site, self.config.data_loader.num_of_sites,
                                                     alpha=alpha, scale=scale, reuse=False, name='sites')


            w1 = self.config.model.cosface_site_period_weight
            w2 = self.config.model.cosface_period_weight
            w3 = self.config.model.cosface_site_weight
            return w1*cls_loss + w2*period_loss + w3*site_loss
        return coss_loss1

    def weighted_coss_loss_wrapper(self, alpha, scale):
        def coss_loss1(y_true, y_pred):
            y_true_cls = K.argmax(y_true,axis=1)
            num_cls = self.config.data_loader.num_of_classes
            labels_probabilty = y_true[:,:num_cls]

            cls_loss = models.cos_face_loss.weighted_cos_loss(y_pred, y_true_cls, num_cls,labels_probabilty,
                                                     alpha=alpha, scale=scale, reuse=False, name='cls')

            return cls_loss
        return coss_loss1

    def triplet_loss_wrapper(self, margin, is_squared):
        def triplet_loss1(y_true, y_pred):
            # !!! y_true and y_pred must have the smae shape !!!
            # therefore we need to crop label array to be in size (B,)
            #batch_size = K.shape(y_true)[0]
            y_true_ = y_true[:,0]
            if self.config.model.batch_type == 'hard':
                loss, hardest_positive_dist, hardest_negative_dist = models.triplet_loss.batch_hard_triplet_loss(y_true_, y_pred, margin, is_squared)
                return loss
            elif self.config.model.batch_type == 'all':
                if self.config.model.is_batch_all_consider_only_200:
                    loss, fraction_positive_triplets = models.triplet_loss.batch_all_triplet_loss(y_true_, y_pred, margin, is_squared,True)
                else:
                    loss, fraction_positive_triplets = models.triplet_loss.batch_all_triplet_loss(y_true_, y_pred,
                                                                                                  margin, is_squared,
                                                                                                  False)
                return loss
            else:
                'unrecognized batch type'
                raise
        return triplet_loss1

    def triplet_loss_wrapper_batch_hard_hardest_pos_dist(self, margin, is_squared):
        def hardest_pos_dist(y_true, y_pred):
            # !!! y_true and y_pred must have the smae shape !!!
            # therefore we need to crop label array to be in size (B,)
            #batch_size = K.shape(y_true)[0]
            y_true_ = y_true[:,0]
            loss, hardest_positive_dist, hardest_negative_dist = models.triplet_loss.batch_hard_triplet_loss(y_true_, y_pred, margin, is_squared)
            return hardest_positive_dist
        return hardest_pos_dist

    def triplet_loss_wrapper_batch_hard_hardest_neg_dist(self, margin, is_squared):
        def hardest_neg_dist(y_true, y_pred):
            # !!! y_true and y_pred must have the smae shape !!!
            # therefore we need to crop label array to be in size (B,)
            #batch_size = K.shape(y_true)[0]
            y_true_ = y_true[:,0]
            loss, hardest_positive_dist, hardest_negative_dist = models.triplet_loss.batch_hard_triplet_loss(y_true_, y_pred, margin, is_squared)
            return hardest_negative_dist
        return hardest_neg_dist

    def triplet_loss_wrapper_batch_all_positive_fraction(self, margin, is_squared):
        def positive_fraction(y_true, y_pred):
            # !!! y_true and y_pred must have the smae shape !!!
            # therefore we need to crop label array to be in size (B,)
            #batch_size = K.shape(y_true)[0]
            y_true_ = y_true[:,0]
            if self.config.model.is_batch_all_consider_only_200:
                loss, fraction_positive_triplets = models.triplet_loss.batch_all_triplet_loss(y_true_, y_pred, margin,
                                                                                          is_squared,True)
            else:
                loss, fraction_positive_triplets = models.triplet_loss.batch_all_triplet_loss(y_true_, y_pred, margin,
                                                                                          is_squared,False)
            return fraction_positive_triplets
        return positive_fraction


