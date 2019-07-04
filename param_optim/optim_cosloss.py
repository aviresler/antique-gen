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
        MAX_ITR = 150
        test_file = 'test_cosface_bg_removed'
        param_csv_path = test_file + '.csv'
        params_start_col = 8
        loss_col = 5
        if not os.path.exists('params_jsons'):
            os.mkdir('params_jsons')

        for kk in range(MAX_ITR):

            # releasing memory
            K.clear_session()
            # @todo: find cleaner solution, instead of waiting for memory release
            # current solution: downgrade keras to version 2.1.6
            #time.sleep(10)

            # read basic config

            args = get_args()
            config = process_config(args.config)


            # update config
            config, experiment = update_config(config, param_csv_path, params_start_col, loss_col, test_file)

            # create the experiments dirs
            config.callbacks.tensorboard_log_dir += experiment
            config.callbacks.checkpoint_dir += experiment
            create_dirs([config.callbacks.tensorboard_log_dir , config.callbacks.checkpoint_dir])


            if int(experiment) == -1:
                break

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

            # updating results in csv file
            with open(param_csv_path, 'r') as f3:
                r = csv.reader(f3)
                lines = list(r)

            lines[int(experiment)+1][4] =  '{0:.3f}'.format(trainer.val_loss[-1])
            lines[int(experiment) + 1][5] = '{0:.3f}'.format(trainer.loss[-1])
            lines[int(experiment) + 1][6] = str(len(trainer.loss))
            lines[int(experiment) + 1][7] = time.strftime("%Y-%m-%d",time.localtime())

            accuracy, period_accuracy, site_accuracy = trainer.get_accuracy()

            lines[int(experiment) + 1][1] = '{0:.3f}'.format(accuracy)
            lines[int(experiment) + 1][2] = '{0:.3f}'.format(period_accuracy)
            lines[int(experiment) + 1][3] = '{0:.3f}'.format(site_accuracy)


            with open(param_csv_path, 'w') as f4:
                writer = csv.writer(f4)
                writer.writerows(lines)



    except Exception as e:
        print(e)
        sys.exit(1)

def update_config(config, param_csv_path, params_start_col, loss_col, test_file):

    # get needed run and its params
    cnt = 0
    experiment = -1
    with open(param_csv_path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if cnt == 0:
                print(row)
                param_names = row[params_start_col:]
                print(param_names)
            if cnt > 0:
                loss = row[loss_col]
                if loss == '-1':
                    experiment = row[0]
                    print('experiment ' + experiment)
                    if int(experiment) == -1:
                        break

                    params = row[params_start_col:]

                    # update config dictionary
                    for i, param in enumerate(param_names):
                        print(param + ' ' + params[i])
                        param_list = param.split('.')
                        try:
                            config[param_list[0]][param_list[1]] = float(params[i])
                        except:
                            config[param_list[0]][param_list[1]] = params[i]


                    # update params that are valid only for param optim runs
                    config['callbacks']['is_save_model'] = False
                    if not str.startswith(config['data_loader']['data_dir_train'], '../'):
                        config['data_loader']['data_dir_train'] = '../' + config['data_loader']['data_dir_train']
                        config['model']['classes_confusion_prior'] = '../' + config['model']['classes_confusion_prior']
                        config['data_loader']['data_dir_valid'] = '../' + config['data_loader']['data_dir_valid']
                        config['data_loader']['data_dir_train_test'] = '../' + config['data_loader']['data_dir_train_test']
                        config['data_loader']['data_dir_valid_test'] = '../' + config['data_loader']['data_dir_valid_test']
                        config['data_loader']['classes_info_csv_file'] = '../' + config['data_loader'][
                            'classes_info_csv_file']

                    # save run json for future inquiries
                    json1 = json.dumps(config.toDict(), indent=4)
                    f2 = open('params_jsons/' + test_file + '_' + experiment + '.json', "w")
                    f2.write(json1)
                    f2.close()

                    break
                else:
                    continue

            cnt = cnt + 1

    return config, experiment

if __name__ == '__main__':
    main()


