# Deep Antique
A deep learning model for predictive archaeology and archaeological community detection.

# Prediction of archaeological period / site. 
In order to try our pretrained model on your query image, follow the following steps (no further installation is required):
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
Antique images, and their period_site labels were taken from the following website: http://www.antiquities.org.il/t/default_en.aspx

The images were grouped together according to their period_site label.
For example, all images of artifacts from Me\`arat ha-Gulgolet site that were dated to Middle Palaeolithic
period were grouped together to a single class.

A thumbnail version of the images, split into classes, can be downloaded from:
https://drive.google.com/file/d/1V8Zdr6tAdm_QoEk39BcYRupoHYI2PoL2/view?usp=sharing

Each train/valid folder has 200 sub-folders, from '0' to '199', where each sub-folder represent a class.
Class name can be seen in the data_loader/classes_top200.csv info file.

For example, folder '37' correspond to (\<period>_\<site>): Middle Palaeolithic_Me`arat Tannur, as can be seen in data_loader/classes_top200.csv info file:

![alt text](https://github.com/aviresler/antique-gen/blob/master/misc/data_info.png)

In order to get the full resolution images (up to about 600 pixels width/height), for research purpose, please contact us: aviresler@gmail.com 


# Training
train.json configuration file is placed in configs folder. Update the train/valid data paths and run:

train.py -c configs/train.json

# Pretrained weights
Pretrained weights can be found in: 
https://drive.google.com/file/d/1hyxQOjT2-CPcjWx4H8X0mUM6kicoMHYr/view?usp=sharing

# Embeddings
Training/Validation set embeddings (.csv file), labels (.tsv file) and images' file names (.csv file)  can be found in
the following links:

Single model: https://drive.google.com/drive/folders/1Ri9RY0PrgNGANnBjy6NdZ2OymMXUMFkO?usp=sharing

Ensemble of 5 models: https://drive.google.com/drive/folders/1seuwH-q9zBTfwgKRRj5tHU8VnAlBL2uj?usp=sharing

* Embeddings are in .csv file, where they have shape of (num_of_images x embedding length): 
  8031x1536 for the training set, 1878x1536 for validation set.
  
* Labels correspond to folder names (also class id data_loader/classes_top200.csv info file).

* In order to reproduce the reported accuracy, run get_valid_nearest_neighbor.py, after placing embeddings/labels in:
'evaluator/embeddings/', 'evaluator/labels/' folders.

* In order to get training/validation sets embeddings, run: get_embeddings_config.py -c configs/train.json,
with the option to start with the pretrained weights in pretraind_model row in configs/train.json.


# Community detection
Run get_communities method in evaluator/communities/community_wrapper.py - example for that can be found in 
community_wrapper.py, in the bottom of the script. 
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

