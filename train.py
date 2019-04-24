from utils.config import process_config
from utils.dirs import create_dirs
from utils.args import get_args
from utils import factory
from data_loader.default_generator import get_default_generator
from data_loader.default_generator import get_testing_generator
from models.cosloss_model import CosLosModel
from trainers.cosloss_trainer import CosLossModelTrainer
import sys
import csv
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import numpy as np
import json
import os
import time
from evaluator.get_valid_nearest_neighbor import eval_model
from data_loader.triplet_data_loader import TripletGenerator
from data_loader.cosface_data_loader import CosFaceGenerator
import tensorflow as tf
from keras import backend as K
import time



def main():
    # capture the config path from the run arguments
    # then process the json configuration fill
    try:

        # read basic config
        args = get_args()
        config = process_config(args.config)

        # create the experiments dirs
        create_dirs([config.callbacks.tensorboard_log_dir, config.callbacks.checkpoint_dir])

        print('train generator')
        datagen_args = dict(rotation_range=config.data_loader.rotation_range,
                            width_shift_range=config.data_loader.width_shift_range,
                            height_shift_range=config.data_loader.height_shift_range,
                            shear_range=config.data_loader.shear_range,
                            zoom_range=config.data_loader.zoom_range,
                            horizontal_flip=config.data_loader.horizontal_flip)

        if config.model.loss == 'cosface':
            train_generator = CosFaceGenerator(config, datagen_args, True)
            valid_generator = CosFaceGenerator(config, datagen_args, False)
        elif config.model.loss == 'triplet':
            train_generator = TripletGenerator(config, datagen_args, True)
            valid_generator = TripletGenerator(config, datagen_args, False)
        else:
            print('invalid loss type')
            raise


        print('Create the model.')
        model = CosLosModel(config)

        print('loading pretrained model')
        if not config.model.pretraind_model == 'None':
            model.load(config.model.pretraind_model)

        print('Create the trainer')
        trainer = CosLossModelTrainer(model.model, train_generator ,valid_generator, config)

        print('Start training the model.')
        trainer.train()

        trainer.get_accuracy()





    except Exception as e:
        print(e)
        sys.exit(1)



if __name__ == '__main__':
    main()


