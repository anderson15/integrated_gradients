
# Illustrating the use of discrete integrated gradients using Titanic data and XGBoost.

# Developed using Python 3.10.9 and XGBoost 1.51.
# Confirmed still works on Python 3.12.9 and XGBoost 2.1.1.

import numpy as np
import pandas as pd
import xgboost as xgb

working_dir = ""

# Load the integrated gradient functions:
exec(open(working_dir + "/code_python/integrated_gradient_functions.py").read())

# ---------------------------
# Prepare Titanic data.
# ---------------------------

# Download the Titanic data: https://www.kaggle.com/c/titanic/data
# The test data doesn't have labels so don't use that part.

# Variables:
#    pclass = ticket class (1, 2, or 3)
#    sex = sex
#    age = age in years
#    sibsp = number of siblings or spouses aboard
#    parch = number of parents / children aboard
#    ticekt = ticket number
#    fare = passenger fare
#    cabin = cabin number
#    embarked = port of embarkation (C, Q, or S)

full = pd.read_csv(working_dir + "/data/titanic_train.csv")
full.rename(columns={"Survived":"survived", "Pclass":"pclass","Age":"age", "SibSp":"sibsp","Parch":"parch", "Fare":"fare"}, inplace=True)

# Two observations are missing embarked:
full.dropna(subset=["Embarked"], inplace=True)

# 177 are missing age, impute and add indicator for missing:
miss = pd.isnull(full.age)
full["age_miss"] = miss.astype(int)
full.loc[miss, "age"] = 30

full["male"] = (full.Sex == "male").astype(int)

full["embark_c"] = (full.Embarked == "C").astype(int)
full["embark_q"] = (full.Embarked == "Q").astype(int)
full["embark_s"] = (full.Embarked == "S").astype(int)

full = full[["survived", "pclass", "age", "age_miss", "sibsp", "parch", "fare", "male", "embark_c", "embark_q", "embark_s"]];

y = full["survived"]
full.drop("survived", inplace=True, axis=1)
x = full.values
y = y.values

# ---------------------------
# Train XGBoost model.
# ---------------------------

# I divided titanic_train.csv into training, validation, and test subsets and searched for hyperparameters. 
param = {'eta':0.457, 'gamma':3.0, 'max_depth':20, 'subsample':0.8, 'lambda':0, 'alpha':0, 'colsample_bytree':0.4, 'booster':'gbtree', 'objective':'binary:logistic', 'eval_metric':'logloss'}
num_round = 17
dx = xgb.DMatrix(x, label=y);
bst = xgb.train(param, dx, num_round)

# ---------------------------
# Explanations.
# ---------------------------

# Run this section plugging in different values for a_ind and b_ind to compare
# the feature contributions for different pairs of passengers.
a_ind = 9
b_ind = 19

a = x[a_ind, :]
b = x[b_ind, :]

# This vector shows roughly how much each feature contributed to the difference in 
# model output at point A versus point B:
feature_contributions = discrete_ig(a, b, 1000, bst, "xgboost")

# To see the change in the probability of survival if you switch each feature value from A to B:
y_hat = bst.predict(xgb.DMatrix(np.array([a, b])))
feat_names = ["pclass", "age", "age_miss", "sibsp", "parch", "fare", "male", "embark_c", "embark_q", "embark_s","prob_survived"]
dat = pd.DataFrame(index=feat_names)
dat["A"] = np.append(a, y_hat[0])
dat["B"] = np.append(b, y_hat[1])
dat["f_cont"] = np.append(feature_contributions, 0.0)
print(dat)

# The sum of the feature contributions will approximate the difference in predictions. Experimenting 
# with different (A, B) points, you will see that the quality of the approximation varies.
print(y_hat[1] - y_hat[0])
print(np.sum(feature_contributions))


