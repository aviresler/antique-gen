from base.base_model import BaseModel
from keras.layers import Dense, Dropout
from keras import applications
from keras.models import Model
from keras import regularizers
from keras import optimizers

class SoftMaxModel(BaseModel):
    def __init__(self, config):
        super(SoftMaxModel, self).__init__(config)
        self.build_model()

    def build_model(self):

        self.model = applications.inception_resnet_v2.InceptionResNetV2(include_top=True, weights='imagenet',
                                                                   input_shape=(self.config.model.img_width, self.config.model.img_height, 3))

        self.model.layers.pop()
        x = self.model.layers[-1].output
        x = Dense(200, activation="relu", kernel_regularizer=regularizers.l2(0.0001))(x)
        x = Dropout(0.5)(x)
        predictions = Dense(200, activation="softmax")(x)
        self.model = Model(input=self.model.input, output=predictions)

        adam1 = optimizers.Adam(lr=0.0)


        self.model.compile(
              loss='categorical_crossentropy',
              optimizer=adam1,
              metrics=['accuracy'])
