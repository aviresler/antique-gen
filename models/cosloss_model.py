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
                x = Dense(int(self.config.model.embedding_dim), activation="relu", kernel_regularizer=regularizers.l2(0.0001))(x)
            else:
                x = Dense(int(self.config.model.embedding_dim), kernel_regularizer=regularizers.l2(0.0001))(x)

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


        if self.config.model.loss == 'triplet':
            loss_func = self.triplet_loss_wrapper(self.config.model.margin, self.config.model.is_squared)
        elif self.config.model.loss == 'cosface':
            loss_func = self.coss_loss_wrapper(self.config.model.alpha, self.config.model.scale)
        else:
            print('invalid loss type')
            raise


        self.model.compile(
              loss=loss_func,
              optimizer=adam1)

    def coss_loss_wrapper(self, alpha, scale):
        def coss_loss1(y_true, y_pred):
            y_true_casted = K.cast(y_true, dtype='int32')
            y_true_casted_ = y_true_casted[:, 0]
            #y_true_casted = K.argmax(y_true_casted, axis= 1)
            #y_true_casted = self.model.input[1]
            return models.cos_face_loss.cos_loss(y_pred, y_true_casted_, self.config.data_loader.num_of_classes, alpha=alpha, scale=scale, reuse=True)
        return coss_loss1

    def triplet_loss_wrapper(self, margin, is_squared):
        def triplet_loss1(y_true, y_pred):
            # !!! y_true and y_pred must have the smae shape !!!
            # therefore we need to crop label array to be in size (B,)
            #batch_size = K.shape(y_true)[0]
            y_true_ = y_true[:,0]
            #y_true_casted = K.cast(y_true, dtype='int32')
            # y_true_casted = K.expand_dims(y_true_casted, axis=-1)
            # print(K.int_shape(y_true_casted))
            return models.triplet_loss.batch_hard_triplet_loss(y_true_, y_pred, margin, is_squared)
        return triplet_loss1
