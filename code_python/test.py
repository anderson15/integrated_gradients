
# This script tests the logic of the discrete integrated gradient code.

# Instead of fitting a model to data, I use an analytical data generating process (DGP).
# I use a simple function for the DGP because that allows me to compute the true line 
# integrals. I compare the actual integrals with the approximated integrated gradients.

import random
import numpy as np

working_dir = ""

# Load the integrated gradient functions:
exec(open(working_dir + "/code_python/integrated_gradient_functions.py").read())

# Define the DGP:
def data_generating_process(x_array):
    n = x_array.shape[0]
    out = np.zeros(n)
    for i in range(n): 
        x = x_array[i,0]
        y = x_array[i,1]
        out[i] = x + x**2 + x*y
    return out

# Run and rerun the following to compute the integrated gradients over different paths.
# You can also experiment with the number of steps.

num_steps = 100
start_pt = np.random.normal(0,1,2)
end_pt = np.random.normal(0,1,2)
approx_contributions = discrete_ig(start_pt, end_pt, num_steps, data_generating_process, "analytic")

# True feature contributions (line integrals) computed analytically:
a = start_pt[0]
b = start_pt[1]
c = end_pt[0]
d = end_pt[1]
scale = end_pt - start_pt
out_0 = scale[0] * (a + b/2 + c + d/2 + 1)
out_1 = scale[1] * (a + 0.5*(c - a))
true_contributions = np.array([out_0, out_1])

print('  True feature contributions: ', true_contributions)
print('Approx feature contributions: ', approx_contributions)

# The sum of the feature contributions approximates the difference in predictions:
start_and_end_func_values = data_generating_process(np.array([start_pt, end_pt]))
dif = start_and_end_func_values[1] - start_and_end_func_values[0]
print("  Difference in predictions: ", dif)
print("  Sum of true contributions: ", sum(true_contributions))
print("Sum of approx contributions: ", sum(approx_contributions))
