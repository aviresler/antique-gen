
from keras.layers import Dense, Dropout, GlobalAveragePooling2D, Input, ZeroPadding2D, concatenate
from keras.layers.core import Dense, Dropout, Activation,Flatten, Reshape, Lambda
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.engine.topology import Layer
import keras.backend as K
import keras


def vgg_attention(inp):

    # block 1
    x = ZeroPadding2D((1, 1))(inp)
    x = Conv2D(64, (3, 3), activation='relu', name='block1_conv1')(x)
    x = ZeroPadding2D((1, 1))(x)
    x = Conv2D(64, (3, 3), activation='relu', name='block1_conv2')(x)

    # block 2
    x = ZeroPadding2D((1, 1))(x)
    x = Conv2D(128, (3, 3), activation='relu', name='block2_conv1')(x)
    x = ZeroPadding2D((1, 1))(x)
    x = Conv2D(128, (3, 3), activation='relu', name='block2_conv3')(x)

    # block 3
    x = ZeroPadding2D((1, 1))(x)
    x = Conv2D(256, (3, 3), activation='relu', name='block3_conv1')(x)
    x = ZeroPadding2D((1, 1))(x)
    x = Conv2D(256, (3, 3), activation='relu', name='block3_conv2')(x)
    x = ZeroPadding2D((1, 1))(x)
    local1 = Conv2D(256, (3, 3), activation='relu', name='block3_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(local1)

    # block 4
    x = ZeroPadding2D((1, 1))(x)
    x = Conv2D(512, (3, 3), activation='relu', name='block4_conv1')(x)
    x = ZeroPadding2D((1, 1))(x)
    x = Conv2D(512, (3, 3), activation='relu', name='block4_conv2')(x)
    x = ZeroPadding2D((1, 1))(x)
    local2 = Conv2D(512, (3, 3), activation='relu', name='block4_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(local2)

    # block 5
    x = ZeroPadding2D((1, 1))(x)
    x = Conv2D(512, (3, 3), activation='relu', name='block5_conv1')(x)
    x = ZeroPadding2D((1, 1))(x)
    x = Conv2D(512, (3, 3), activation='relu', name='block5_conv2')(x)
    x = ZeroPadding2D((1, 1))(x)
    local3 = Conv2D(512, (3, 3), activation='relu', name='block5_conv3')(x)
    #local3 = Conv2D(512, (3, 3), name='block5_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(local3)

    #g = GlobalAveragePooling2D(name='g')(x)

    # #Add Fully Connected Layer
    x = ZeroPadding2D((1, 1))(x)
    x = Conv2D(512, (3, 3), activation='relu', name='block6_conv1')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)
    x = ZeroPadding2D((1, 1))(x)
    x = Conv2D(512, (3, 3), activation='relu', name='block6_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)
    x = Reshape([int(x.shape[1]) * int(x.shape[2]) * int(x.shape[3])])(x)
    g = Dense(512, name='g')(x)  # batch*512

    return (g, local1, local2, local3)

def att_block( g, local1, local2, local3,outputclasses, regularizer, loss_type = 'softmax' ):

    l1 = Dense(512, kernel_regularizer=regularizer, name='l1connectordense')(local1)  # batch*x*y*512
    c1 = ParametrisedCompatibility(kernel_regularizer=regularizer, name='cpc1')([l1, g])  # batch*x*y
    flatc1 = Reshape([int(c1.shape[1]) * int(c1.shape[2])], name='flatc1')(c1)
    a1 = Activation('softmax', name='softmax1')(flatc1)  # batch*xy
    reshaped1 = Reshape([int(l1.shape[1]) * int(l1.shape[2]), 512], name='reshape1')(l1)  # batch*xy*512.
    g1 = Lambda(lambda lam: K.squeeze(K.batch_dot(K.expand_dims(lam[0], 1), lam[1]), 1), name='g1')(
        [a1, reshaped1])  # batch*512.

    l2 = local2
    c2 = ParametrisedCompatibility(kernel_regularizer=regularizer, name='cpc2')([l2, g])
    flatc2 = Reshape([int(c2.shape[1]) * int(c2.shape[2])], name='flatc2')(c2)
    a2 = Activation('softmax', name='softmax2')(flatc2)
    reshaped2 = Reshape([int(l2.shape[1]) * int(l2.shape[2]), 512], name='reshape2')(l2)  # batch*xy*512.
    g2 = Lambda(lambda lam: K.squeeze(K.batch_dot(K.expand_dims(lam[0], 1), lam[1]), 1), name='g2')([a2, reshaped2])

    l3 = local3
    c3 = ParametrisedCompatibility(kernel_regularizer=regularizer, name='cpc3')([l3, g])
    flatc3 = Reshape([int(c3.shape[1]) * int(c3.shape[2])], name='flatc3')(c3)
    a3 = Activation('softmax', name='softmax3')(flatc3)
    reshaped3 = Reshape([int(l3.shape[1]) * int(l3.shape[2]), 512], name='reshape3')(l3)  # batch*xy*512.
    g3 = Lambda(lambda lam: K.squeeze(K.batch_dot(K.expand_dims(lam[0], 1), lam[1]), 1), name='g3')([a3, reshaped3])

    glist = [g3]
    glist.append(g2)
    glist.append(g1)
    predictedG = concatenate([glist[0], glist[1], glist[2]],name='predictedG', axis=1)
    #xp = Dense(328, kernel_regularizer=regularizer, name='embeddings1')(predictedG)
    x = Dense(outputclasses, kernel_regularizer=regularizer, name=str(outputclasses) + 'ConcatG')(predictedG)
    if loss_type == 'softmax':
        out = Activation("softmax", name='out')(x)
    else:
        out = predictedG
    return out, Reshape([int(l1.shape[1]), int(l1.shape[1])],name='alpha')(a1),predictedG

class ParametrisedCompatibility(Layer):

    def __init__(self, kernel_regularizer=None, **kwargs):
        super(ParametrisedCompatibility, self).__init__(**kwargs)
        self.regularizer = kernel_regularizer

    def build(self, input_shape):
        self.u = self.add_weight(name='u', shape=(input_shape[0][3], 1), initializer='uniform',
                                 regularizer=self.regularizer, trainable=True)
        super(ParametrisedCompatibility, self).build(input_shape)

    def call(self, x):  # add l and g. Dot the sum with u.
        return K.dot(K.map_fn(lambda lam: (lam[0] + lam[1]), elems=(x), dtype='float32'), self.u)

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][1], input_shape[0][2])