from utils.config import process_config
from utils.dirs import create_dirs
from utils.args import get_args
from utils import factory
from data_loader.default_generator import get_testing_generator
from models.cosloss_model import CosLosModel
import sys
import csv
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import numpy as np
import json
import os
import time
from evaluator.get_valid_nearest_neighbor import test_model_on_query_imgages, get_prediction_string
import tensorflow as tf
from keras import backend as K
import time
import numpy as np
from lime import lime_image
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt


def main():
    # capture the config path from the run arguments
    # then process the json configuration fill
    try:

        # read basic config
        args = get_args()
        config = process_config(args.config)

        print('test generator')
        test_generator = get_testing_generator(config, folder=config.data_loader.query_images_folder)

        print('Create the model.')
        model = CosLosModel(config)

        print('loading pretrained model')
        if not config.model.pretraind_model == 'None':
            model.load(config.model.pretraind_model)

        # lime interpret
        explainer = lime_image.LimeImageExplainer()

        label_map = (test_generator.class_indices)
        label_map = dict((v, k) for k, v in label_map.items())  # flip k,v
        train_embeddings = np.genfromtxt(config.data_loader.train_embbedings_csv, delimiter=',')
        train_labels = np.genfromtxt(config.data_loader.train_labels_tsv, delimiter='\t')



        for k in range(len(test_generator)):
            img, y_true_ = test_generator.__getitem__(k)
            y_true = [label_map[x] for x in y_true_]

            img = np.squeeze(img)
            y_true = int(y_true[0])

            predict_fn = predict_wrapper(train_embeddings,train_labels,model,y_true, config.data_loader.num_of_classes,
                                         class_mode='site_period',isPrint= False,classes_csv_file= config.data_loader.classes_info_csv_file)
            prob = predict_fn(img[np.newaxis,:])
            prob = np.squeeze(prob)
            description = get_prediction_string(prob,y_true,classes_csv_file= config.data_loader.classes_info_csv_file,class_mode='site_period')
            print(k)
            print(description)


            arg_sort_probability = np.argsort(prob)
            arg_sort_probability = np.flip(arg_sort_probability)


            explanation = explainer.explain_instance(img, predict_fn, top_labels=5, hide_color=0, num_samples=1000)

            temp, mask = explanation.get_image_and_mask(arg_sort_probability[0], positive_only=True, num_features=5, hide_rest=False)

            folder,file_name = test_generator.filenames[k].split('/')
            folder = config.data_loader.output_folder + '/' + folder
            if not os.path.exists(folder):
                os.mkdir(folder)

            f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
            ax1.get_xaxis().set_visible(False)
            ax1.get_yaxis().set_visible(False)
            ax2.axis([0, 10, 0, 10])


            ax1.imshow(mark_boundaries(temp / 2 + 0.5, mask))
            ax1.title.set_text('query image and significant pixels for the top prediction')
            ax2.text(0.2,3,description,fontsize=14)

            plt.savefig(folder + '/' + file_name)


    except Exception as e:
        print(e)
        sys.exit(1)

    return 0



def predict_wrapper(train_embeddings,train_labels,model,query_label,num_of_classes,
                    classes_csv_file = '',class_mode = 'site_period', isPrint= False):
    def predict(img):
        query_embeddings = model.model.predict(img)
        probability = test_model_on_query_imgages(train_embeddings,train_labels,query_embeddings,query_label,num_of_classes, class_mode=class_mode,
                                              isPrint=isPrint, classes_csv_file=classes_csv_file )
        return probability
    return predict

if __name__ == '__main__':
    main()


