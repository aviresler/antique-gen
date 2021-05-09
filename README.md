# Deep Antique
A deep learning model for predictive archaeology and archaeological community detection.

# Prediction of archaeological period / site. 
In order to try our pretrained model on your query image, follow the following steps:
* Open "antique_pedia.ipynb", and press the following icon to launch it on the colab platform:

![alt text](https://github.com/aviresler/antique-gen/blob/master/misc/colab.png)
* Follow the instructions above each cell, and run them sequentially.
* On the last cell, you will be requested to upload query image of an artifact, which can be in the following formats: jpg/png/bmp etc.
* Wait few seconds until the image is uploaded, and until the model performs inference.
* 5 similar images with their period_site label will be shown at the bottom of the cell. For example, here is a query image and its most similar image from our database:

![alt text](https://github.com/aviresler/antique-gen/blob/master/misc/query_pred0.png)


# Installation
Create a conda environment based on environment.yaml file:

* conda env create --file environment.yaml
* conda activate tf-gpu-2

# Data

# Train
train.json configuration file is placed in configs folder. Update the train/valid data paths and run:

train.py -c configs/train.json

# Community detection
Run get_communities method in evaluator/communities/community_wrapper.py.
It will generate communities based on modularity maximization, from model predictions.
For each validation-set image, up to 50 training-set nearest neighbours are taken into account.

    Args:
        - num_of_neighbors (int): number of neighbors to be considered, number between 1-50.
        is_self_loops (bool): Whether to form a graph which has edges between nodes to themselves.
        - relevant_period_groups (list of int): period groups that should be considered when forming graph. if -1, all periods
        are taken into account. The list of period groups is in classes_csv_file, at priod_group_column.
        - full_confusion_csv (str): path to csv file with the confusion data.
        - classes_csv_file (str): path to csv file with the classes data.
        - priod_group_column (str): relevant column for period_groups in classes_csv_file
    
    The function saves community_summary.csv file, and returns a string with the summary.

For example, here is a part of the summary for certain parameters:

![alt text](https://github.com/aviresler/antique-gen/blob/master/misc/community_detection.png)
