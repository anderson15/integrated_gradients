# Integrated Gradients
This repository provides code for computing integrated gradients, a method for explaining the output of machine learning models ([Sundararajan, Taly, & Yan, 2017](https://arxiv.org/abs/1703.01365)).

The unique feature of this code is that it allows computing integrated "gradients" for non-differentiable functions, e.g., XGBoost models. The method only requires function evaluations; there is no need to supply a gradient. I call this the **discrete integrated gradient** method. In addition, this repository provides code for standard integrated gradients, i.e., for differentiable functions such as neural networks. Other options for the standard method include: [ankurtaly](https://github.com/ankurtaly/Integrated-Gradients) and [TensorFlow core](https://www.tensorflow.org/tutorials/interpretability/integrated_gradients). 

This repository includes code for the:

* Python language:
  * Discrete integrated gradients for [XGBoost](https://github.com/dmlc/xgboost)
  * Standard integrated gradients for [Tensorflow via Keras](https://keras.io/)
* Julia language:
  * Discrete integrated gradients for [EvoTrees.jl](https://github.com/Evovest/EvoTrees.jl) and [XGBoost.jl](https://github.com/dmlc/XGBoost.jl)
  * Standard integrated gradients for [Flux.jl](https://github.com/FluxML/Flux.jl)

The code can be easily altered for any programming language or machine learning model. The **example** files in this repository demonstrate the use of the code. 

For both the discrete and standard methods, the only **hyperparameter** is the number of steps for approximating the integral. You might start with 100, then use more steps if you need more accuracy and fewer if you need more speed. The files called **test** compare the discrete integrated gradient to analytically computed line integrals. You can use the test files to see how the discrete approximation converges to the analytical solution as the number of steps increases.

The code is released under the MIT license.

If you publish work using this code, please cite this [article in Neural Computing and Applications:](https://doi.org/10.1007/s00521-023-08597-8). If you don't have access to that journal, please see this [open-access, view-only version]( https://rdcu.be/dbo4S). 
```
BibTeX format:
@article{Anderson2023,
  author   = {Anderson, Andrew A.},
  journal  = {Neural Computing and Applications},
  title    = {Testing machine learning explanation methods},
  year     = {2023},
  issn     = {1433-3058},
  month    = {May},
   doi     = {10.1007/s00521-023-08597-8},
  refid    = {Anderson2023},
  url      = {https://doi.org/10.1007/s00521-023-08597-8},
}

RIS format:
TY  - JOUR
AU  - Anderson, Andrew A.
PY  - 2023
DA  - 2023/05/04
TI  - Testing machine learning explanation methods
JO  - Neural Computing and Applications
SN  - 1433-3058
UR  - https://doi.org/10.1007/s00521-023-08597-8
DO  - 10.1007/s00521-023-08597-8
ID  - Anderson2023
ER  - 
```
