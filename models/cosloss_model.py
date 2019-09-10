from base.base_model import BaseModel
from keras.layers import Dense, Dropout, Conv2D, Activation, MaxPooling2D, Flatten, GlobalAveragePooling2D, Input,MaxPool2D
from keras.models import Sequential
from keras import applications
from keras.models import Model
from keras import regularizers
from keras import optimizers
from keras import backend as K
import models.cos_face_loss
import models.triplet_loss
import numpy as np
from models.vgg_attention import vgg_attention, att_block
import keras
import tensorflow as tf
from keras_efficientnets import EfficientNetB3
from tensorflow.contrib.opt import AdamWOptimizer
from models.adamW import AdamW
from models.spatial_transformer import SpatialTransformer

from models.STN.utils import get_initial_weights
from models.STN.layers import BilinearInterpolation


class CosLosModel(BaseModel):
    def __init__(self, config):
        super(CosLosModel, self).__init__(config)
        self.regularizer = keras.regularizers.l2(self.config.model.weight_decay)
        self.metrics = []
        self.loss_weights = []
        self.loss_func = []
        self.build_model()

    def build_model(self):

        # build net
        if self.config.model.type == "inceptionResnetV2":
            self.build_inceptionResNetV2()
        elif self.config.model.type == "vgg":
            self.build_vgg()
        elif self.config.model.type == "vgg_attention":
            self.build_vgg_attention()
        elif self.config.model.type == "efficientNet":
            self.build_efficientNet()
        elif self.config.model.type == "efficientNetSTN":
            self.build_efficientNetSTN()
        elif self.config.model.type == "dummy":
            self.build_dummy_model()
        else:
            print('model type is not supported')
            raise

        # configure optimizer
        if self.config.trainer.learning_rate_schedule_type == 'LearningRateScheduler':
            lr_ = 0.0
            # adam1 = optimizers.Adam(lr=0.0)
        elif self.config.trainer.learning_rate_schedule_type == 'ReduceLROnPlateau':
            print('decrease_platue')
            lr_ = self.config.trainer.learning_rate
            # adam1 = optimizers.Adam(lr=self.config.trainer.learning_rate)
        else:
            print('invalid learning rate configuration')
            raise

        adam1 = AdamW(lr=lr_, beta_1=0.9, beta_2=0.999, epsilon=None, decay=self.config.trainer.learning_rate_decay,
                      weight_decay=self.config.model.weight_decay, batch_size=self.config.data_loader.batch_size,
                      samples_per_epoch=401, epochs=30)

        # configure loss and metrics
        if self.config.model.loss == 'triplet':
            self.configure_triplet_loss()
        elif self.config.model.loss == 'cosface' or self.config.model.loss == 'softmax':
            self.configure_cosface_or_softmax_loss()
        else:
            print('invalid loss type')
            raise

        self.model.compile(
              loss=self.loss_func,
              loss_weights=self.loss_weights,
              optimizer=adam1,
              metrics=self.metrics
        )

    # localization network for the STN
    def loc_net(self,input_shape):
        #print(input_shape)
        b = np.zeros((2, 3), dtype='float32')
        b[0, 0] = 1
        b[1, 1] = 1
        w = np.zeros((50, 6), dtype='float32')
        weights = [w, b.flatten()]

        loc_input = Input(input_shape)

        locnet = MaxPool2D(pool_size=(2, 2))(loc_input)
        locnet = Conv2D(20, (5, 5))(locnet)
        locnet = MaxPool2D(pool_size=(2, 2))(locnet)
        locnet = Conv2D(20, (5, 5))(locnet)
        locnet = Flatten()(locnet)
        locnet = Dense(50)(locnet)
        locnet = Activation('relu')(locnet)
        #weights = get_initial_weights(50)
        locnet = Dense(6, weights=weights)(locnet)


        return locnet

    def build_inceptionResNetV2(self):
        self.model = applications.inception_resnet_v2.InceptionResNetV2(include_top=True, weights='imagenet',
                                                                        input_shape=(self.config.model.img_width,
                                                                                     self.config.model.img_height, 3))
        self.model.layers.pop()
        x = self.model.layers[-1].output

        if self.config.model.is_use_relu_on_embeddings:
            x = Dense(int(self.config.model.embedding_dim), activation="relu", kernel_regularizer=self.regularizer)(x)
        else:
            x = Dense(int(self.config.model.embedding_dim), kernel_regularizer=self.regularizer)(x)

        x = Dropout(0.5, name='embeddings')(x)
        self.model = Model(input=self.model.input, output=x)

    def build_vgg(self):
        base_model = applications.vgg16.VGG16(include_top=False, weights='imagenet',
                                              input_shape=(
                                                  self.config.model.img_width, self.config.model.img_height, 3))
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        embeddings = Dense(int(self.config.model.embedding_dim), kernel_regularizer=self.regularizer, name='embeddings')(x)
        x = Dropout(0.5)(embeddings)
        x = Dense(int(self.config.data_loader.num_of_classes), kernel_regularizer=self.regularizer)(x)
        out = Activation("softmax", name='out')(x)
        self.model = Model(inputs=base_model.input, outputs=[embeddings, out])

    def build_dummy_model(self):
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

    def build_vgg_attention(self):
        inp = Input(shape=(self.config.model.img_width, self.config.model.img_height, 3), name='main_input')
        (g, local1, local2, local3) = vgg_attention(inp)
        out, alpha, G = att_block(g, local1, local2, local3, int(self.config.data_loader.num_of_classes), self.regularizer)
        self.model = Model(inputs=inp, outputs=[g, out, alpha])

    def build_efficientNet(self):
        base_model = EfficientNetB3((self.config.model.img_width, self.config.model.img_height, 3),
                                    include_top=False, weights='imagenet')
        inp = Input(shape=(self.config.model.img_width, self.config.model.img_height, 3), name='main_input')
        x = base_model(inp)

        if self.config.model.num_of_fc_at_net_end == 2:
            x = GlobalAveragePooling2D(name='gpa_f')(x)
            x = Dropout(0.5)(x)
            embeddings = Dense(int(self.config.model.embedding_dim), name='embeddings')(x)
            x = Dense(int(self.config.data_loader.num_of_classes), )(embeddings)
            out = Activation("softmax", name='out')(x)
        else:
            embeddings = GlobalAveragePooling2D(name='embeddings')(x)
            x = Dense(int(self.config.data_loader.num_of_classes), )(embeddings)
            out = Activation("softmax", name='out')(x)

        self.model = Model(inputs=inp, outputs=[embeddings, out])

    def build_efficientNetSTN(self):
        base_model = EfficientNetB3((self.config.model.img_width, self.config.model.img_height, 3),
                                    include_top=False, weights='imagenet')
        inp = Input(shape=(self.config.model.img_width, self.config.model.img_height, 3), name='main_input')
        locnet = MaxPool2D(pool_size=(2, 2))(inp)
        locnet = Conv2D(100, (5, 5))(locnet)  # 20
        locnet = MaxPool2D(pool_size=(4, 4))(locnet)  # 2, 2
        locnet = Conv2D(200, (5, 5))(locnet)  # 20
        # locnet = MaxPool2D(pool_size=(2, 2))(locnet)
        locnet = Flatten()(locnet)
        locnet = Dense(80)(locnet)  # 50
        locnet = Activation('relu')(locnet)
        weights = get_initial_weights(80)  # 50
        locnet = Dense(6, weights=weights)(locnet)
        stn = BilinearInterpolation((300, 300), name='stn')([inp, locnet])
        x = base_model(stn)

        if self.config.model.num_of_fc_at_net_end == 2:
            x = GlobalAveragePooling2D(name='gpa_f')(x)
            x = Dropout(0.5)(x)
            embeddings = Dense(int(self.config.model.embedding_dim), name='embeddings')(x)
            x = Dense(int(self.config.data_loader.num_of_classes), )(embeddings)
            out = Activation("softmax", name='out')(x)
        else:
            embeddings = GlobalAveragePooling2D(name='embeddings')(x)
            x = Dense(int(self.config.data_loader.num_of_classes), )(embeddings)
            out = Activation("softmax", name='out')(x)

        self.model = Model(inputs=inp, outputs=[stn, embeddings, out])

    def configure_triplet_loss(self):
        self.loss_func = self.triplet_loss_wrapper(self.config.model.margin, self.config.model.is_squared)
        if self.config.model.batch_type == 'hard':
            hardest_pos_dist = self.triplet_loss_wrapper_batch_hard_hardest_pos_dist(self.config.model.margin,
                                                                                     self.config.model.is_squared)
            self.metrics.append(hardest_pos_dist)
            hardest_neg_dist = self.triplet_loss_wrapper_batch_hard_hardest_neg_dist(self.config.model.margin,
                                                                                     self.config.model.is_squared)
            self.metrics.append(hardest_neg_dist)
        if self.config.model.batch_type == 'all':
            pos_fraction = self.triplet_loss_wrapper_batch_all_positive_fraction(self.config.model.margin,
                                                                                 self.config.model.is_squared)
            self.metrics.append(pos_fraction)
        self.loss_weights = {"embeddings": 1.0, "out": 0.0}

    def configure_cosface_or_softmax_loss(self):
        loss_func_cos = self.coss_loss_wrapper(self.config.model.alpha, self.config.model.scale)
        loss_func_softmax = self.softmax_loss_wrapper()
        loss_func_empty = self.empty_loss_wrapper()

        if self.config.model.type == "efficientNetSTN":
            self.metrics = {"stn": loss_func_empty, "embeddings ": loss_func_empty, "out": "acc"}
            self.loss_func = {"stn": loss_func_empty, "embeddings": loss_func_cos, "out": loss_func_softmax}
            if self.config.model.loss == 'cosface':
                self.loss_weights = {"stn": 0.0, "embeddings": 1.0, "out": 0.0}
            else:
                self.loss_weights = {"stn": 0.0, "embeddings": 0.0, "out": 1.0}
        elif self.config.model.type == "efficientNet" or self.config.model.type == "vgg":
            self.metrics = {"embeddings ": loss_func_empty, "out": "acc"}
            self.loss_func = {"embeddings": loss_func_cos, "out": loss_func_softmax}
            if self.config.model.loss == 'cosface':
                self.loss_weights = {"embeddings": 1.0, "out": 0.0}
            else:
                self.loss_weights = {"embeddings": 0.0, "out": 1.0}
        elif self.config.model.type == "inceptionResnetV2":
            self.loss_func = loss_func_cos
            self.loss_weights = {"embeddings": 1.0}
        elif self.config.model.type == "vgg_attention":
            self.loss_weights = {"g": 0.0, "out": 1.0, "alpha": 0.0}
            self.loss_func = {"g": loss_func_cos, "out": loss_func_softmax, "alpha": loss_func_empty}
            self.metrics = {"g": loss_func_empty, "out": "acc", "alpha": loss_func_empty}



    def softmax_loss_wrapper(self):
        def softmax_loss1(y_true, y_pred):
            y_true_casted = K.cast(y_true, dtype='int32')
            y_true_cls = y_true_casted[:, 0]
            return K.sparse_categorical_crossentropy(y_true_cls,y_pred)
        return softmax_loss1

    def empty_loss_wrapper(self):
        def softmax_loss1(y_true, y_pred):
            return K.constant(0)
        return softmax_loss1

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


