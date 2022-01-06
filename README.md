# Automobile-Price-Pediction

This pipeline trains a linear regressor to predict a car's price based on technical features such as make, model, horsepower, and size etc. Because we are trying to answer the question "How much?" this is called a regression problem. However, you can apply the same fundamental steps in this example to tackle any type of machine learning problem whether it be regression, classification, clustering, and so on.

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
Machine learning problems vary. Common machine learning tasks include classification, clustering, regression, and recommender systems, each of which might require a different algorithm. Your choice of algorithm often depends on the requirements of the use case. After  pick an algorithm, we need to tune its parameters to train a more accurate model.  then we need to evaluate all models based on metrics like accuracy.

Since the goal of this sample is to predict automobile prices, and because the label column (price) is continuous data, a regression model can be a good choice. We use Linear Regression for this pipeline.

Use the Split Data module to randomly divide the input data so that the training dataset contains 80% of the original data and the testing dataset contains 20% of the original data.
# Test, evaluate, and compare
Split the dataset and use different datasets to train and test the model to make the evaluation of the model more objective.After the model is trained, you can use the Score Model and Evaluate Model modules to generate predicted results and evaluate the models and compared with diferent Regression algorithms.

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

 1) import pandas as pd
 2) import numpy as np
 3) import matplotlib.pyplot as plt
 4) import seaborn as sns

# Data Visulization ans Anaysis

1) symboling : Its assigned insurance risk rating,A value of +3 indicates that the auto is risky,-3 that it is probably pretty safe.
![symb](https://user-images.githubusercontent.com/95012573/146322800-449ea447-e669-4188-8e57-1e0b5d6c2e94.PNG)

2)body_style : Hardtop and convertible are the most expensive whereas hatchbacks are the cheapest.
![car body](https://user-images.githubusercontent.com/95012573/146322950-9cbe190b-508a-4d9a-b74c-48a9f47e4d29.PNG)

3)fule_type- diesel engine is used more compare to gas engine. 
![fule](https://user-images.githubusercontent.com/95012573/146323008-2eda81c5-de1b-47d2-9f96-d6507039fcfa.PNG)

4)make: toyota has highest number of vehicales compare to all other compnays . Toyota, a Japanese company has the most no of models.
![make](https://user-images.githubusercontent.com/95012573/146323064-3723bdee-221a-41bc-a57a-891c13efc5c6.PNG)

5) according to automobile the num-of-cylinders and engine size is described. For example, according to the attribute, "four" has 156.0 counts, the mean value of this column is 112.980769 ,maximum value is 156.
![g](https://user-images.githubusercontent.com/95012573/146323103-4545d8c6-9f5e-423f-beb2-cd14165be2f1.PNG)

# Conclusion
The Linear regression model is compared with other regression model and it gives good result for linear regression model.
Comparing these three models, we conclude that the MLR model is the best model to be able to predict price from our dataset. This result makes sense, since we have 27 variables in total, and we know that more than one of those variables are potential predictors of the final car price.




