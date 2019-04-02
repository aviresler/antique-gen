from base.base_model import BaseModel
from keras.layers import Dense, Dropout
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

        self.model = applications.inception_resnet_v2.InceptionResNetV2(include_top=True, weights='imagenet',
                                                                   input_shape=(self.config.model.img_width, self.config.model.img_height, 3))

        self.model.layers.pop()
        x = self.model.layers[-1].output

        if self.config.model.is_use_relu_on_embeddings:
            x = Dense(200, activation="relu", kernel_regularizer=regularizers.l2(0.0001))(x)
        else:
            x = Dense(200, kernel_regularizer=regularizers.l2(0.0001))(x)

        x = Dropout(0.5)(x)

        self.model = Model(input=self.model.input, output=x)

        adam1 = optimizers.Adam(lr=0.0)

        if self.config.model.is_use_triplet_loss:
            loss_func = self.triplet_loss_wrapper(self.config.model.margin, self.config.model.is_squared)
        else:
            loss_func = self.coss_loss_wrapper(self.config.model.alpha, self.config.model.scale)


        self.model.compile(
              loss=loss_func,
              optimizer=adam1)

    def coss_loss_wrapper(self, alpha, scale):
        def coss_loss1(y_true, y_pred):
            y_true_casted = K.cast(y_true, dtype='int32')
            # y_true_casted = K.expand_dims(y_true_casted, axis=-1)
            # print(K.int_shape(y_true_casted))
            return models.cos_face_loss.cos_loss(y_pred, y_true_casted, 200, alpha=alpha, scale=scale, reuse=True)
        return coss_loss1

    def triplet_loss_wrapper(self, margin, is_squared):
        def triplet_loss1(y_true, y_pred):
            y_true_casted = K.cast(y_true, dtype='int32')
            # y_true_casted = K.expand_dims(y_true_casted, axis=-1)
            # print(K.int_shape(y_true_casted))
            return models.triplet_loss.batch_hard_triplet_loss(y_true_casted, y_pred, margin, is_squared)
        return triplet_loss1
