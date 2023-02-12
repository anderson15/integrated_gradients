
# This script tests the logic of the discrete integrated gradient code.

# Instead of fitting a model to data, I use an analytical data generating process (DGP).
# I use a simple function for the DGP because that allows me to compute the true line 
# integrals. I compare the actual integrals with the approximated integrated gradients.

using Random

working_dir = ""

# Load the integrated gradient functions:
include(working_dir * "/code_julia/integrated_gradient_functions.jl")

# Define the DGP:
function data_generating_process(x_array)
    n = size(x_array)[1]
    out = zeros(n)
    for i = 1:n
        x = x_array[i,1]
        y = x_array[i,2]
        out[i] = x + x^2 + x*y
    end
    return out
end

# Run and rerun the following function to compute the integrated gradients over different paths.
# Also experiment with the number of steps.

function test(num_steps)
    start_pt = randn(2);
    end_pt = randn(2);
    approx_contributions = discrete_ig(start_pt, end_pt, num_steps, data_generating_process, "analytic");
    # True feature contributions (line integrals) computed analytically:
    a = start_pt[1];
    b = start_pt[2];
    c = end_pt[1];
    d = end_pt[2];
    scale = end_pt - start_pt;
    out_1 = scale[1] * (a + b/2 + c + d/2 + 1);
    out_2 = scale[2] * (a + 0.5*(c - a));
    true_contributions = vcat(out_1, out_2);
    println("  True feature contributions: ", true_contributions)
    println("Approx feature contributions: ", approx_contributions)
    # The sum of the feature contributions approximates the difference in predictions:
    start_and_end_func_values = data_generating_process(permutedims(hcat(start_pt, end_pt)));
    dif = start_and_end_func_values[2] - start_and_end_func_values[1];
    println("  Difference in predictions: ", dif)
    println("  Sum of true contributions: ", sum(true_contributions))
    println("Sum of approx contributions: ", sum(approx_contributions))
end

test(100)

