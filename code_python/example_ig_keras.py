
# Illustrating the use of integrated gradients with Tensorflow via Keras using Titanic data.

# Developed using Python 3.10.9 and Tensorflow 2.10.0.

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import xception

working_dir = ""

# Load the integrated gradient functions:
exec(open(working_dir + "/code_python/integrated_gradient_functions.py").read())

# ---------------------------
# Data preparation.
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
# Train Keras model.
# ---------------------------

feat_names = ["pclass", "age", "age_miss", "sibsp", "parch", "fare", "male", "embark_c", "embark_q", "embark_s"]

# Normalize features:
x_norm = np.zeros(x.shape)
num_cols = len(feat_names)
for i in range(num_cols):
    ave = x[:,i].mean()
    std = x[:,i].std()
    print(i, ave, std)
    if std > 0:
       x_norm[:,i] = (x[:,i] - ave) / std

# I divided titanic_train.csv into training, validation, and test subsets and obtained the following
# architecture / hyperparameters. 
model = keras.Sequential()
model.add(layers.Dense(units=25, input_shape=[num_cols], activation='relu'))
model.add(layers.Dense(units=10, input_shape=[25], activation='relu'))
model.add(layers.Dense(units=10, input_shape=[10], activation='relu'))
model.add(layers.Dense(units=1, input_shape=[10], activation='sigmoid'))
learn_rate = 0.5
model.compile(loss=tf.keras.losses.MeanSquaredError(), optimizer=tf.keras.optimizers.legacy.SGD(learning_rate=learn_rate), metrics=['accuracy'])
model.fit(x_norm, y, epochs=200, batch_size=660, verbose=1, shuffle=True)

# ---------------------------
# Explanations.
# ---------------------------

# Run this section plugging in different values for a_ind and b_ind to compare
# the feature contributions for different pairs of passengers.
a_ind = 0
b_ind = 1

a = x_norm[a_ind, :]
b = x_norm[b_ind, :]

# This vector shows roughly how much each feature contributed to the difference in 
# model output at point A versus point B:
feature_contributions = integrated_gradients(a, b, 1000, model)

# To see the change in the probability of survival if you switch each feature value from A to B:
y_hat = model(np.array([a, b])).numpy().flatten()
feat_names = ["pclass", "age", "age_miss", "sibsp", "parch", "fare", "male", "embark_c", "embark_q", "embark_s","prob_survived"]
dat = pd.DataFrame(index=feat_names)
dat["A"] = np.append(x[a_ind,:], y_hat[0])
dat["B"] = np.append(x[b_ind,:], y_hat[1])
dat["f_cont"] = np.append(feature_contributions, 0.0)
print(dat)

# The sum of the feature contributions will approximate the difference in predictions. Experimenting 
# with different (A, B) points, you will see that the quality of the approximation varies.
print(y_hat[1] - y_hat[0])
print(np.sum(feature_contributions))



