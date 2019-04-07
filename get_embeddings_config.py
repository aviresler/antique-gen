from utils.config import process_config
from utils.dirs import create_dirs
from utils.args import get_args
from utils import factory
import sys
from data_loader.default_generator import get_testing_generator
import numpy as np
from evaluator.get_valid_nearest_neighbor import eval_model

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
        model = factory.create("models."+config.model.name)(config)

        print('loading pretrained model')
        model.load(config.model.pretraind_model)

        generators = [train_generator,valid_generator]
        generators_id = ['_train', '_valid']


        for m, generator in enumerate(generators):
            num_of_images = len(generator.filenames)
            labels = np.zeros((num_of_images, 1), dtype=np.int)
            predication = np.zeros((num_of_images, config.model.embedding_dim), dtype=np.float32)

            label_map = (generator.class_indices)
            label_map = dict((v, k) for k, v in label_map.items())  # flip k,v

            print(len(generator))
            for k in range(len(generator)):
                print(k)
                x, y_true = next(generator)
                y_pred = model.model.predict(x)
                y_true = [label_map[x] for x in y_true]

                if k == len(generator) -1:
                    labels[k*config.model.batch_size:,0] = y_true
                    predication[k*config.model.batch_size:, :] = y_pred
                else:
                    labels[k*config.model.batch_size:(k+1)*config.model.batch_size,0] = y_true
                    predication[k*config.model.batch_size:(k+1)*config.model.batch_size, :] = y_pred

            if m == 0:
                train_labels = labels
                train_prediction = predication
            else:
                valid_labels = labels
                valid_prediction = predication


            np.savetxt('evaluator/labels/' + config.exp.name + generators_id[m] + '.tsv', labels, delimiter=',')
            np.savetxt('evaluator/embeddings/' + config.exp.name + generators_id[m] + '.csv', predication, delimiter=',')

        eval_model(train_prediction,valid_prediction,train_labels, valid_labels, config.exp.name)




    except Exception as e:
        print(e)
        sys.exit(1)

if __name__ == '__main__':
    main()
