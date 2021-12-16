# Automobile-Price-Pediction
Learn how to build a machine learning regression model without writing a single line of code using the designer.

This pipeline trains a linear regressor to predict a car's price based on technical features such as make, model, horsepower, and size. Because you're trying to answer the question "How much?" this is called a regression problem. However, you can apply the same fundamental steps in this example to tackle any type of machine learning problem whether it be regression, classification, clustering, and so on.

The fundamental steps of a training machine learning model are:

1) Get the data
2) Pre-process the data
3) Train the model
4) Evaluate the model
# Get the data
This sample uses the Automobile price data (Raw) dataset, which is from DATAMITES course center . The dataset contains 26 columns that contain information about automobiles, including make, model, price, vehicle features (like the number of cylinders), MPG, and an insurance risk score. The goal of this sample is to predict the price of the car.
# Preprocessing the data
The main data preparation tasks include data cleaning, integration, transformation.encoding the categorical features , Use the Select Columns in Dataset module to exclude normalized-losses that have many missing values. Then use Clean Missing Data to remove the rows that have missing values. This helps to create a clean set of training data.
# Training the model
Machine learning problems vary. Common machine learning tasks include classification, clustering, regression, and recommender systems, each of which might require a different algorithm. Your choice of algorithm often depends on the requirements of the use case. After you pick an algorithm, you need to tune its parameters to train a more accurate model. You then need to evaluate all models based on metrics like accuracy, intelligibility, and efficiency.

Since the goal of this sample is to predict automobile prices, and because the label column (price) is continuous data, a regression model can be a good choice. We use Linear Regression for this pipeline.

Use the Split Data module to randomly divide the input data so that the training dataset contains 80% of the original data and the testing dataset contains 20% of the original data.
# Test, evaluate, and compare
Split the dataset and use different datasets to train and test the model to make the evaluation of the model more objective.
After the model is trained, you can use the Score Model and Evaluate Model modules to generate predicted results and evaluate the models.and compared with diferent Regression algorithms.

It is divided into four parts:

1) Data Wrangling

Pre processing data in python
Dealing missing values
Exploratory Data Analysis

2) Descriptive statistics
Groupby
Analysis of variance
value_counts
Data visulization

3) Model Development

Simple Linear Regression
Decision Tree Regression 
Random Forest Regression
XGBoost Regression
Model Review and Evaluation

# Softwares and Libraries Used:
- Anaconda Distribution
- Jupyter Notebook

- Numpy
- Pandas
- Matplotlib
- Seaborn

# Importing the Modules:

 import pandas as pd
 import numpy as np
 import matplotlib.pyplot as plt
 import seaborn as sns

