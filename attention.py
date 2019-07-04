# -*- coding: utf-8 -*-

import keras
from keras.optimizers import SGD
from keras.engine.topology import Layer
from keras.layers import Input, ZeroPadding2D, concatenate
from keras.layers.core import Dense, Dropout, Activation,Flatten, Reshape, Lambda
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.models import Model
import keras.backend as K
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import ModelCheckpoint
from keras.models import model_from_json
import numpy as np
import math
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.utils import class_weight
import os
#import cv2
from data_loader.default_generator import get_default_generator
from utils.args import get_args
from utils.config import process_config
from utils.dirs import create_dirs

with_local = 'False' # use extra loss with zoom-in information?

#os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def vgg_attention(inp):

    # block 1
    x = ZeroPadding2D((1, 1))(inp)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = ZeroPadding2D((1, 1))(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)

    # block 2
    x = ZeroPadding2D((1, 1))(x)
    x = Conv2D(128, (3, 3), activation='relu')(x)
    x = ZeroPadding2D((1, 1))(x)
    x = Conv2D(128, (3, 3), activation='relu')(x)

    # block 3
    x = ZeroPadding2D((1, 1))(x)
    x = Conv2D(256, (3, 3), activation='relu')(x)
    x = ZeroPadding2D((1, 1))(x)
    x = Conv2D(256, (3, 3), activation='relu')(x)
    x = ZeroPadding2D((1, 1))(x)
    local1 = Conv2D(256, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(local1)

    # block 4
    x = ZeroPadding2D((1, 1))(x)
    x = Conv2D(512, (3, 3), activation='relu')(x)
    x = ZeroPadding2D((1, 1))(x)
    x = Conv2D(512, (3, 3), activation='relu')(x)
    x = ZeroPadding2D((1, 1))(x)
    local2 = Conv2D(512, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(local2)

    # block 5
    x = ZeroPadding2D((1, 1))(x)
    x = Conv2D(512, (3, 3), activation='relu')(x)
    x = ZeroPadding2D((1, 1))(x)
    x = Conv2D(512, (3, 3), activation='relu')(x)
    x = ZeroPadding2D((1, 1))(x)
    local3 = Conv2D(512, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(local3)

    # Add Fully Connected Layer
    x = ZeroPadding2D((1, 1))(x)
    x = Conv2D(512, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)
    x = ZeroPadding2D((1, 1))(x)
    x = Conv2D(512, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)
    x = Reshape([int(x.shape[1]) * int(x.shape[2]) * int(x.shape[3])])(x)
    g = Dense(512, activation='relu', name='g')(x)  # batch*512

    return (g, local1, local2, local3)

def att_block(g, local1, local2, local3, outputclasses):
    weight_decay = 0.0005
    regularizer = keras.regularizers.l2(weight_decay)

    l1 = Dense(512, kernel_regularizer=regularizer, name='l1connectordense')(local1)  # batch*x*y*512
    c1 = ParametrisedCompatibility(kernel_regularizer=regularizer, name='cpc1')([l1, g])  # batch*x*y
    flatc1 = Reshape([int(c1.shape[1]) * int(c1.shape[2])],name='flatc1')(c1)
    a1 = Activation('softmax', name='softmax1')(flatc1)  # batch*xy
    reshaped1 = Reshape([int(l1.shape[1]) * int(l1.shape[2]) , 512], name='reshape1')(l1)  # batch*xy*512.
    g1 = Lambda(lambda lam: K.squeeze(K.batch_dot(K.expand_dims(lam[0], 1), lam[1]), 1), name='g1')([a1, reshaped1])  # batch*512.

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
    predictedG = concatenate([glist[0], glist[1], glist[2]], axis=1)
    x = Dense(outputclasses, kernel_regularizer=regularizer, name=str(outputclasses) + 'ConcatG')(predictedG)
    out = Activation("softmax", name='concatsoftmaxout')(x)

    return out, Reshape([int(l1.shape[1]),int(l1.shape[1])])(a1)

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

if __name__ == '__main__':

    args = get_args()
    config = process_config(args.config)

    batch_size = 8
    img_rows, img_cols = 224, 224 # Resolution of inputs
    num_channels = 3
    num_classes = 200
    nb_epoch = 50

    training_generator = get_default_generator(config, True)
    validation_generator = get_default_generator(config, False)

    x,y = training_generator.__getitem__(0)
    print(x.shape)
    print(y.shape)
    print(x.dtype)
    print(y.dtype)

    #print(np.sort(np.unique(training_generator.classes)))
    sorted_labels = np.sort(training_generator.classes)
    class_weight = class_weight.compute_class_weight('balanced', np.unique(sorted_labels), sorted_labels)

    inp = Input(shape=(224, 224, num_channels), name='main_input')
    (g, local1, local2, local3) = vgg_attention(inp)
    out, alpha = att_block(g, local1, local2, local3, num_classes)
    model = Model(inputs=inp, outputs=out)

    #print(model.summary())

    #model.load_weights('weights-improvement_vgg-26-0.87.hdf5', by_name=True)

    sgd = SGD(lr=1e-3, decay=1e-7, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])


    filepath = "weights-improvement_vgg_att_l1_1e-4-{epoch:02d}-{val_acc:.2f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min', save_weights_only=True)
    weights_save_epoch = checkpoint

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-6)
    callbacks_list = [weights_save_epoch, reduce_lr]

    # Train model on dataset
    model.fit_generator(generator=training_generator,
                        validation_data=validation_generator,
                        epochs=nb_epoch,
                        shuffle=True,
                        use_multiprocessing=True,
                        max_queue_size=10,
                        workers=5,
                        verbose=1,
                        class_weight=class_weight,
                        callbacks=callbacks_list)

    # Save the model architecture
    with open('vgg16_model_attention_l1_1e-4.json', 'w') as f:
        f.write(model.to_json())

    # Save the weights
    model.save_weights('vgg16_weights_attention_l1_1e-4.h5')

    # Make predictions
    predictions_valid = model.predict_generator(generator=validation_generator, use_multiprocessing=True, workers=5, verbose=1)

    val_trues = np.array(validation_generator.ind_labels)
    val_trues = val_trues[::num_channels]
    val_trues_cut = val_trues[:-(len(val_trues) % batch_size)]
    val_pred = np.argmax(predictions_valid, axis=1)
    # confusion matrix
    print('Confusion Matrix: ')
    cm = confusion_matrix(val_trues_cut, val_pred)
    np.set_printoptions(threshold=10000)
    #print(cm)
    # metrics calc
    precisions, recall, f1_score, _ = precision_recall_fscore_support(val_trues_cut, val_pred)
    print('Average precision is: ')
    print(np.ndarray.mean(precisions))
    print('Average recall is: ')
    print(np.ndarray.mean(recall))
    print('Average f1_score is: ')
    print(np.ndarray.mean(f1_score))

    print('attention - l1 loss, 1e-4')