
# Illustrating the use of discrete integrated gradients using Titanic data and EvoTrees.

# Runs using Julia 1.11.3, and EvoTrees 0.17.0.

using DataFrames
using CSV
using EvoTrees
using EvoTrees: fit
using Random

working_dir = ""

# Load the integrated gradient functions:
include(working_dir * "/code_julia/integrated_gradient_functions.jl")

# -------------------------------------
# Prepare Titanic data.
# -------------------------------------

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

full = CSV.read(working_dir * "/data/titanic_train.csv", DataFrame);

rename!(full, :Survived => :survived);
rename!(full, :Pclass => :pclass);
rename!(full, :Age => :age);
rename!(full, :SibSp => :sibsp);
rename!(full, :Parch => :parch);
rename!(full, :Fare => :fare);

# Two are missing embarked:
full = full[.!ismissing.(full.Embarked),:];

# 177 are missing age, impute and add indicator for missing:
full.age_miss =convert(Array{Float64,1},  ismissing.(full.age));
full.age[ismissing.(full.age)] .= 30.0;

full.male = convert(Array{Float64, 1}, full.Sex .== "male");

full.embark_c = convert(Array{Float64, 1}, full.Embarked .== "C");
full.embark_q = convert(Array{Float64, 1}, full.Embarked .== "Q");
full.embark_s = convert(Array{Float64, 1}, full.Embarked .== "S");

full = full[:, [:survived, :pclass, :age, :age_miss, :sibsp, :parch, :fare, :male, :embark_c, :embark_q, :embark_s]];

# Convert data to array:
y = full.survived;
x = Matrix(full[:, [:pclass, :age, :age_miss, :sibsp, :parch, :fare, :male, :embark_c, :embark_q, :embark_s]]);
x = identity.(x); # to get rid of Union missing

# -------------------------------------
# Train EvoTrees model.
# -------------------------------------

# I divided titanic_train.csv into training, validation, and test subsets and searched for hyperparameters. 

model_type = "evotrees"
config = EvoTreeRegressor(loss = :logloss,
    nrounds = 26,
    max_depth = 15,
    eta = 0.127,
    rowsample = 0.784,
    colsample = 0.785,
    alpha = 0.0,
    lambda = 0.0,
    gamma = 2.62);
model = fit(config; x_train=x, y_train=y, verbosity=1);

# -------------------------------------
# Explanations.
# -------------------------------------

# Run this section plugging in different values for a_ind and b_ind to compare
# the feature contributions for different pairs of passengers.
a_ind = 10;
b_ind = 20;

a = x[a_ind, :];
b = x[b_ind, :];

# This vector shows roughly how much each feature contributed to the difference in 
# model output at point A versus point B:
feature_contributions = discrete_ig(a, b, 1000, model, model_type);

# To see the change in the probability of survival if you switch each feature value from A to B:
y_hat = model(permutedims(hcat(a,b)));
row_names = ["pclass", "age", "age_miss", "sibsp", "parch", "fare", "male", "embark_c", "embark_q", "embark_s","prob_survived"];
dat = DataFrame(Dict(:x => row_names));
dat[!, "A"] = vcat(a, y_hat[1]);
dat[!, "B"] = vcat(b, y_hat[2]);
dat[!, "f_cont"] = vcat(feature_contributions, 0.0);
println(dat)

# The sum of the feature contributions will approximate the difference in predictions. Experimenting 
# with different (A, B) points, you will see that the quality of the approximation varies.
println(y_hat[2] - y_hat[1])
println(sum(feature_contributions))



