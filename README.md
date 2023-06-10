# Integrated Gradients
This repository provides code for computing integrated gradients, a method for explaining the output of machine learning models ([Sundararajan, Taly, & Yan, 2017](https://arxiv.org/abs/1703.01365)).

The unique feature of this code is that it allows computing integrated "gradients" for non-differentiable functions, e.g., XGBoost models. The method only requires function evaluations; there is no need to supply a gradient. I call this the **discrete integrated gradient** method. In addition, this repository provides code for standard integrated gradients, i.e., for differentiable functions such as neural networks. Other options for the standard method include: [ankurtaly](https://github.com/ankurtaly/Integrated-Gradients) and [TensorFlow core](https://www.tensorflow.org/tutorials/interpretability/integrated_gradients). 

If you publish work using this code, please cite this [article in Neural Computing and Applications](https://doi.org/10.1007/s00521-023-08597-8). If you don't have access to that journal, please see this [open-access, view-only version]( https://rdcu.be/dbo4S). The citation in BibTeX and RIS formats is provided at the bottom of this page.

## Description of Code
This repository includes code for the:

* Python language:
  * Discrete integrated gradients for [XGBoost](https://github.com/dmlc/xgboost)
  * Standard integrated gradients for [Tensorflow via Keras](https://keras.io/)
* Julia language:
  * Discrete integrated gradients for [EvoTrees.jl](https://github.com/Evovest/EvoTrees.jl) and [XGBoost.jl](https://github.com/dmlc/XGBoost.jl)
  * Standard integrated gradients for [Flux.jl](https://github.com/FluxML/Flux.jl)

The code can be easily altered for any programming language or machine learning model. The **example** files in this repository demonstrate the use of the code. 

For both the discrete and standard methods, the only **hyperparameter** is the number of steps to use when approximating the integral. You might start with 100, then use more steps if you need more accuracy or fewer if you need more speed. The files called **test** compare the discrete integrated gradient to analytically computed line integrals. You can use the test files to see how the discrete approximation converges to the analytical solution as the number of steps increases.

The code is released under the MIT license.

## Summary of Method
Our goal is to explain the difference in a function's value at two distinct points. The method of integrated gradients is based on line integrals. Instead of integrating the function itself, we integrate its gradient. Recall that, by the second fundamental theorem of calculus, the integral of a function's derivative between two points is equal to the difference in the function's value at those two points. Since we want the total change broken down by the contribution of each input variable, we integrate each component of the gradient vector.

I illustrate using a simple function that's amenable to analytic integration: $f(x,y)=x^{2}+x+xy$. Suppose we want to understand the difference in the function's value at the point $(c,d)$, compared to its value at the baseline $(a,b)$. We parameterize the path of integration as: $r(t)=(1-t)\langle a,b\rangle+t\langle c,d\rangle$ for $0\leq t\leq1$, which simplifies to: $r(t)=\langle(1-t)a+tc,(1-t)b+td\rangle$. Thus we have:
```math
\begin{align*}
x &= (1-t)a+tc \\
y &= (1-t)b+td
\end{align*}
```
The gradient of $f$ consists of the two partial derivatives:
```math
\begin{align*}
\frac{\partial f}{\partial x} &= 2x+1+y \\
\frac{\partial f}{\partial y} &= x
\end{align*}
```
Plugging the parameterization into the partial derivatives and scaling by distance in each dimension (due to the change in variables from the parameterization), the integrals for each partial are:
```math
\begin{align*}
x \text{-component} &=(c-a)\int_{0}^{1}\left(2\left((1-t)a+tc\right)+1+(1-t)b+td\right)dt \\
y \text{-component} &=(d-b)\int_{0}^{1}\left((1-t)a+tc\right)dt
\end{align*}
```
Integrate and simplify to get:
```math
\begin{align*}
x\text{-component} &=(c-a)\left(a+\frac{b}{2}+c+\frac{d}{2}+1\right) \\
y\text{-component} &=(d-b)\left(a+\frac{1}{2}\left(c-a\right)\right)
\end{align*}
```
These formulas show how much of the function's total change can be attributed to the change in $x$ versus the change in $y$. 

Suppose we are integrating from the point $(1,2)$ to the point $(3,4)$. The function values are: $f(1,2)=4$ and $f(3,4)=24$, for an increase of $20$. We decompose this total increase as follows:
```math
\begin{align*}
x\text{-component} &=(3-1)\left(1+\frac{2}{2}+3+\frac{4}{2}+1\right)=16 \\
y\text{-component} &=(4-2)\left(1+\frac{1}{2}(3-1)\right)=4
\end{align*}
```
As expected, the sum of the two components is equal to the function's total change. The components show that the $x$ variable is the more important factor when moving from $(1,2)$ to $(3,4)$.

## Method for Non-Differentiable Functions
The idea behind the discrete integrated gradient method is to define a series of points along a vector from the starting point to the end point. At each point along the path, and for each feature, move that feature one step toward the end point while keeping all other features fixed and record the change in the model's prediction. The sum of these changes constitutes the "integrated gradient" for each feature. 

The standard method of integrated gradients uses the actual gradient but approximates the integral. However, since the gradient is undefined for non-differentiable functions, the discrete integrated gradient method approximates both the gradient and the integral.
Note that more sophisticated finite difference methods are not required (and would provide no benefit). Since we're focused on non-differentiable functions, infinitesimal changes in input variables are generally not associated with any change in the function value.

## Citation
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
