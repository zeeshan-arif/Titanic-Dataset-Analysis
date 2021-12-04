
# Titanic Dataset Analysis

Explore and wrangle the Titanic passenger manifest dataset and developing a predictive model for survival. It is the legendary Titanic ML dataset – the best, first challenge for anyone to dive into Machine Learning and familiarize ourself with how the everything works.

The dataset contains features :-

| Feature        | Description |
| -----------    | ----------- |
| survival       | 0- No or 1- Yes       |
| pclass         | Ticket class        |
| sex            | Sex       |
| Age            | Age in years        |
| sibsp          | Number of siblings / spouses aboard the Titanic       |
| parch          | Number of parents / children aboard the Titanic        |
| ticket         | Ticket number       |
| fare           | Passenger fare        |
| cabin          | Cabin number       |
| embarked       | Port of Embarkation (C = Cherbourg, Q = Queenstown, S = Southampton)        |


## Libraries used

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
```

#### Steps involved -

- Load the dataset using pandas.
- Check the shape of the dataset and whether the dataset is balanced or not.
- Plot a heatmap to check for null values in the dataset.
- Perform EDA on dataset (like distribution of features).
- Fill in the missing values under Age feature and drop the cabin feature because 77% of values are missing and drop the rows where embarked has missing values.
- Convert the categorical features using *get_dummies()*.
- Scale the numerical features using MinMaxScaler.
- Now check for the correlation.
- Since survival feature has two values only either 1 or 0, then it is good to *Logistic Regression*. Check for the accuracy score and confusion matrix.
- In the same way build models using *Decision Tree Classifier, Support Vector Classifier, Random Forest Classifier* and print accuracy score and confusion matrix.
- Now for parameter tuning we use GridSearchCV, and use Logistic Regression, Support Vector Classifier and Random Forest Classifier with it.
