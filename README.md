# AMLS_II_assignment23_24

This project explores the problem of classifiying images of dolphins and whales to identify indivduals. This work 
is based on the **Happywhale - Whale and Dolphin Identification** Kaggle competition.

Data for this competition contains over 50,000 images of over 15,000 unique individual marine mammals from 30 different species collected from 28 different research organizations. 
Individuals have been manually identified and given an individual\_id by marine researchers.

The Kaggle competion had the additional complexity of requiring the ability to classify individuals not included in the training dataset. It was decided to ommit this requirement due to time limitations. The Kaggle competion also measured model performance via Mean Average Precision @ 5 but for this task it was replaced with Accuracy.


## Project Structure

The project is broken up into six folders

* **Classifiers** : Code for PyTorch Neural Networks
* **Datasets** : Contains the cropped training images in a folder called train_images and the labelling data in a file called train.csv
* **Notebooks**: Exploratory Jupyter Notebooks
* **Reference**: Documentation on the assignment task
* **Report**: LaTex files for report
* **Utils**: Holds the custom Torch Dataset class WhaleDataset. Could be expanded to hold other util classes

**main.py** is the entry point to run the training and testing of the classifier. 

The cropped dataset is too large to check in source control. The full dataset will need to be downloaded from

```https://www.kaggle.com/awsaf49/happywhale-data-distribution```



## Run Locally 

Poetry was for dependency management

``` poetry shell ```

``` poetry install ```

``` python main.py ```


Don't have Poetry? I have generated a requirements.txt via 
```poetry export --without-hashes --format=requirements.txt > requirements.txt ``` 

