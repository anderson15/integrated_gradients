
# Illustrating the use of integrated gradients with Flux.jl using Titanic data.

# Developed using Julia 1.8.5 and Flux 0.13.12.
# Still runs with Julia 1.11.3 and Flux 0.16.3.

using DataFrames
using CSV
using Flux
using ForwardDiff
using Random
using StatsBase

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
full_y = full.survived;
full_x = Matrix(full[:, [:pclass, :age, :age_miss, :sibsp, :parch, :fare, :male, :embark_c, :embark_q, :embark_s]]);
full_x = identity.(full_x); # to get rid of Union missing

# ---------------------------
# Train Flux model.
# ---------------------------

# I did hyperparameter tuning elsewhere, now I'll just use the entire dataset.

x_t = transpose(full_x);
y_t = reshape(full_y, 1, length(full_y));

# Normalize features:
x_t_norm = zeros(size(x_t));
num_k = 10 
for i = 1:num_k
    tmp_ave = mean(x_t[i,:]);
    tmp_std =  std(x_t[i,:]);
    x_t_norm[i,:] = (x_t[i,:] .- tmp_ave) ./ tmp_std;
end;

# Convert to Float32: 
x_t_norm = convert(Array{Float32}, x_t_norm);
y_t = convert(Array{Float32}, y_t);

loader = Flux.DataLoader((x_t_norm, y_t), batchsize=889, shuffle=true);

# Specify the model architecture:
model = Chain(
    Dense(num_k, 25, relu),
    Dense(25, 10, relu),
    Dense(10, 10, relu),
    Dense(10, 1, sigmoid)) 

learn_rate = 0.5
optim = Flux.setup(Flux.Descent(learn_rate), model);

# Training loop, using the whole data once for each epoch:
for epoch in 1:200
    for (x, y) in loader
        loss, grads = Flux.withgradient(model) do m
            y_hat = m(x)
            Flux.mse(y_hat, y)
        end
        Flux.update!(optim, model, grads[1])
        #println(loss)
    end
end

# ---------------------------
# Explanations.
# ---------------------------

# While the Python version takes the neural network function as its input,
# the Julia function takes the gradient of that function. This is the gradient
# with respect to the input features, not the gradients used in training.
function g(x)
    out = model(x)[1]
end
function grad_func(x)
    out = ForwardDiff.gradient(g, x)
end

# Run this section plugging in different values for a_ind and b_ind to compare
# the feature contributions for different pairs of passengers.
a_ind = 1
b_ind = 2

a = x_t_norm[:, a_ind]
b = x_t_norm[:, b_ind]

# This vector shows roughly how much each feature contributed to the difference in 
# model output at point A versus point B:
feature_contributions = integrated_gradients(a, b, 1000, grad_func)

# To see the change in the probability of survival if you switch each feature value from A to B:
y_hat = model(x_t_norm[:, [a_ind, b_ind]])
row_names = ["pclass", "age", "age_miss", "sibsp", "parch", "fare", "male", "embark_c", "embark_q", "embark_s","prob_survived"];
dat = DataFrame(Dict(:x => row_names));
dat[!, "A"] = vcat(x_t[:, a_ind], y_hat[1]);
dat[!, "B"] = vcat(x_t[:, b_ind], y_hat[2]);
dat[!, "f_cont"] = vcat(feature_contributions, 0.0);
println(dat)

# The sum of the feature contributions will approximate the difference in predictions. Experimenting 
# with different (A, B) points, you will see that the quality of the approximation varies.
println(y_hat[2] - y_hat[1])
println(sum(feature_contributions))



