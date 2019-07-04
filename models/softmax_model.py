from base.base_model import BaseModel
from keras.layers import Dense, Dropout, GlobalAveragePooling2D, Input, ZeroPadding2D, concatenate
from keras import applications
from keras.models import Model
from keras import regularizers
from keras import optimizers
from models.vgg_attention import vgg_attention, att_block

class SoftMaxModel(BaseModel):
    def __init__(self, config):
        super(SoftMaxModel, self).__init__(config)
        self.build_model()



    def build_model(self):
        if self.config.model.type == 'inceptionResnetV2':
            self.model = applications.inception_resnet_v2.InceptionResNetV2(include_top=True, weights='imagenet',
                                                                       input_shape=(self.config.model.img_width, self.config.model.img_height, 3))
            self.model.layers.pop()
            x = self.model.layers[-1].output
            x = Dense(200, activation="relu", kernel_regularizer=regularizers.l2(0.0001))(x)
            x = Dropout(0.5)(x)
            predictions = Dense(200, activation="softmax")(x)
            self.model = Model(input=self.model.input, output=predictions)
        elif self.config.model.type == 'vgg':
            base_model = applications.vgg16.VGG16(include_top=False, weights='imagenet',
                                                  input_shape=(self.config.model.img_width, self.config.model.img_height, 3))


            x = base_model.output
            x = GlobalAveragePooling2D()(x)
            x = Dense(328, activation='relu')(
                x)  # we add dense layers so that the model can learn more complex functions and classify for better results.
            # x = Dense(1024,activation='relu')(x) #dense layer 2
            # x = Dense(512,activation='relu')(x) #dense layer 3
            preds = Dense(200, activation='softmax')(x)  # final layer with softmax activation

            self.model = Model(inputs=base_model.input, outputs=preds)
            print(self.model.summary())
            #print(model_final.summary())
            #print('hey')
        elif self.config.model.type == 'vgg_attention':
            inp = Input(shape=(224, 224, 3), name='main_input')
            (g, local1, local2, local3) = vgg_attention(inp)
            out, alpha = att_block(g, local1, local2, local3, 200)
            self.model = Model(inputs=inp, outputs=out)
            print(self.model.summary())

            print('here')
        else:
            print('unsupported model type')
            raise

        adam1 = optimizers.Adam(lr=0.0)




        self.model.compile(
              loss='categorical_crossentropy',
              optimizer=adam1,
              metrics=['accuracy'])
