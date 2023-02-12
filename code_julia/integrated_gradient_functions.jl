
# Developed using: 
#  Julia 1.8.5
#  XGBoost 2.2.4
#  EvoTrees 0.14.8
#  Flux 0.13.12

function integrated_gradients(start_pt, end_pt, steps, grad_func)
    dif = end_pt - start_pt
    num_k = length(start_pt)
    # List of points including first point and final point (so steps + 1 number of points):
    path_of_integration = zeros(((steps+1), num_k))
    for i = 1:(steps+1)
        path_of_integration[i,:] = start_pt .+ dif*((i-1)/steps)
    end
    # List of gradients, each corresponds to a point on the path:
    grad_list = zeros(((steps+1), num_k))
    for i = 1:(steps+1)
        tmp_pt = reshape(path_of_integration[i,:], num_k, 1) 
        grad_list[i,:] = grad_func(tmp_pt)
    end
    # Trapezoid approximation of integral:
    grads = zeros((steps, num_k));
    for i = 1:steps
        grads[i,:] = (grad_list[i,:] .+ grad_list[i+1,:]) ./ 2
    end
    ave_grads = mean(grads, dims=1)
    # Account for the distance traveled in each dimension:
    ig = dif .* vec(ave_grads)
    return ig
end 

function discrete_ig(start_pt, end_pt, steps, model, model_type)
    # The model_type argument:"xgboost", "evotrees", "flux",  or "analytic".
    num_k = length(start_pt) 
    dif = end_pt - start_pt
    # Points on the path:
    step_pts = zeros((steps, num_k));
    for i = 1:steps
        step_pts[i,:] = start_pt + dif*(i-1.0) ./ steps
    end
    # Discrete deviations from step_pts for each feature:
    dev_pts = zeros((steps*num_k, num_k));
    inc = (1/steps) * dif
    row_ind = 1
    for s = 1:steps
        for col_ind = 1:num_k
            dev_pts[row_ind, :] = step_pts[s,:]
            dev_pts[row_ind, col_ind] = step_pts[s,col_ind] + inc[col_ind]
            row_ind += 1
        end
    end
    # Append then make predictions:
    dat = vcat(step_pts, dev_pts);
    if model_type == "evotrees"
        y_hat = model(dat)
    end
    if model_type == "xgboost"
        dmat = DMatrix(dat);
        y_hat = predict(model, dmat)
    end
    if model_type == "analytic"
        y_hat = model(dat)
    end

    if model_type == "flux"
        dat = permutedims(dat);
        y_hat = model(dat)
    end
    # Sum the y_hat deltas for each deviation and feature:
    out = zeros(num_k);
    row_ind = steps + 1
    for s = 1:steps
        base_y_hat = y_hat[s]
        for col_ind = 1:num_k
            dev_y_hat = y_hat[row_ind]
            out[col_ind] += dev_y_hat - base_y_hat
            row_ind += 1
        end
    end
    return out
end 

