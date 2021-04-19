# Antique-gen
A deep learning model for predictive archaeology and archaeological community detection.

# Predictions demo
In order to try our pretrained model, follow the following steps:
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

# train