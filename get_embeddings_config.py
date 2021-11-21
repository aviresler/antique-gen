from utils.config import process_config
from utils.dirs import create_dirs
from utils.args import get_args
from utils import factory
import sys
from data_loader.default_generator import get_testing_generator
import numpy as np
from evaluator.get_valid_nearest_neighbor import eval_model
from models.cosloss_model import CosLosModel

def main():
    # capture the config path from the run arguments
    # then process the json configuration fill
    try:
        args = get_args()
        config = process_config(args.config)

        # create the experiments dirs
        create_dirs([config.callbacks.tensorboard_log_dir, config.callbacks.checkpoint_dir])

        print('Create the data generator.')
        train_generator = get_testing_generator(config, True)
        valid_generator = get_testing_generator(config, False)

        print('Create the model.')
        model = CosLosModel(config)

        print('loading pretrained model')
        if not config.model.pretraind_model == 'None':
            model.load(config.model.pretraind_model)
        #model = factory.create("models."+config.model.name)(config)

        print('loading pretrained model')
        model.load(config.model.pretraind_model)

        generators = [train_generator,valid_generator]
        generators_id = ['_train', '_valid']

        for m, generator in enumerate(generators):
            batch_size = config['data_loader']['batch_size']

            num_of_images = len(generator)*(batch_size)
            labels = np.zeros((num_of_images, 1), dtype=np.int)
            predication = np.zeros((num_of_images, int(config.model.embedding_dim)), dtype=np.float32)

            label_map = (generator.class_indices)
            label_map = dict((v, k) for k, v in label_map.items())  # flip k,v

            cur_ind = 0
            for k in range(len(generator)):
                if k % 10 == 0:
                    print(k)
                x, y_true_ = generator.__getitem__(k)
                y_true = [label_map[x] for x in y_true_]
                y_pred = model.model.predict(x)

                # taking only the embeddings output
                y_pred = y_pred[0]

                num_of_items = y_pred.shape[0]

                predication[cur_ind: cur_ind+num_of_items,:] = y_pred
                labels[cur_ind: cur_ind + num_of_items, :] = np.expand_dims(y_true, axis=1)
                cur_ind = cur_ind + num_of_items

            predication = predication[:cur_ind,:]
            labels = labels[:cur_ind, :]

            if m == 0:
                train_labels = labels
                train_prediction = predication
            else:
                valid_labels = labels
                valid_prediction = predication


            np.savetxt('evaluator/labels/' + config.exp.name + generators_id[m] + '.tsv', labels, delimiter=',')
            np.savetxt('evaluator/embeddings/' + config.exp.name + generators_id[m] + '.csv', predication, delimiter=',')

        accuracy = eval_model(train_prediction, valid_prediction, train_labels, valid_labels, config.exp.name, is_save_files= True)
        print('accuracy = {0:.3f}'.format(accuracy))




    except Exception as e:
        print(e)
        sys.exit(1)

if __name__ == '__main__':
    main()
