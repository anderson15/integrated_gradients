# Integrated Gradients
This repository provides code for computing integrated gradients (IGs), a method for explaining the output of machine learning models ([Sundararajan, Taly, & Yan, 2017](https://arxiv.org/abs/1703.01365)).

In addition to standard integrated gradients, I provide code for computing IGs using non-differentiable functions. I call this the **discrete integrated gradient** approach. The code I provide is for XGBoost and EvoTrees, but it's easily altered for any machine learning model. It only requires that you can evaluate the function, there is no need to supply a gradient. 

This repository includes code for the:

* Python language:
  * Discrete Integrated Gradients for [XGBoost](https://github.com/dmlc/xgboost)
  * Regular Integrated Gradients for [Tensorflow via Keras](https://keras.io/)
* Julia language:
  * Discrete Integrated Gradients for [EvoTrees.jl](https://github.com/Evovest/EvoTrees.jl)
  * Regular Integrated Gradients for [Flux.jl](https://github.com/FluxML/Flux.jl)

For both the regular and the discrete IG, the only **hyperparameter** is the number of steps to use when approximating the integral. You might start with 100 then adjust. Use more if you need more accuracy and fewer if you need more speed.

I demonstrate the use of integrated gradients in the **example** files. The files called **test** compare the discrete integrated gradient to analytically computed line integrals. Use it to see how the discrete approximation converges to the analytical solution as the number of steps increases.

The code is released under the MIT license.

If you publish work using this code, please cite this article (obviously this is just a placeholder for now):
```
% BibTeX:
@article{anderson_ig_2023,
  author  = {Andrew A. Anderson},
  journal = {TBD (under review)},
  title   = {Testing Machine Learning Explanation Methods},
  year    = {2023},
}
```
