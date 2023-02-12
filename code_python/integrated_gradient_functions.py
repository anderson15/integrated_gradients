
# Developed using:
#  Python: 3.10.9
#  XGBoost: 1.51 
#  Tensorflow: 2.10.0

def integrated_gradients(start_pt, end_pt, steps, keras_model):
    cols = len(start_pt)
    start_pt = tf.convert_to_tensor(start_pt.reshape(-1, cols))
    end_pt = tf.convert_to_tensor(end_pt.reshape(-1, cols))
    # List of points including first point and final point (so steps + 1 points):
    path_of_integration = [start_pt + (float(i)/steps)*(end_pt-start_pt) for i in range(0, steps+1)]
    # List of gradients, each corresponds to a point:
    grad_list = []
    for i in range(steps+1):
        x_tensor = path_of_integration[i]
        with tf.GradientTape() as t:
            t.watch(x_tensor)
            output = keras_model(x_tensor)
            grad = t.gradient(output, x_tensor)
        grad_list.append(grad[0])
    grads = tf.convert_to_tensor(grad_list, dtype=tf.float64)
    # Trapezoid approximation of integral:
    grads = (grads[:-1] + grads[1:]) / 2.0
    ave_grads = tf.reduce_mean(grads, axis=0)
    # Scale by distance traveled in each dimension:
    ig = (end_pt-start_pt).numpy()*ave_grads.numpy()
    return ig[0,:]

def discrete_ig(start_pt, end_pt, steps, model, model_type):
    # Model type can be "xgboost" or "analytic".
    # Points on the path:
    n_cols = len(start_pt)
    step_pts = [start_pt + (float(i)/steps)*(end_pt-start_pt) for i in range(0,steps)]
    step_pts = np.array(step_pts)
    # Discrete deviations from step_pts for each feature:
    dev_pts = np.zeros((steps*n_cols, len(start_pt)))
    inc = (1/steps) * (end_pt-start_pt)
    row_ind = 0
    for s in range(steps):
        for col_ind in range(n_cols):
            dev_pts[row_ind, :] = step_pts[s,:]
            dev_pts[row_ind, col_ind] = step_pts[s,col_ind] + inc[col_ind]
            row_ind += 1
    dat = np.append(step_pts, dev_pts, axis=0)
    # Make predictions:
    if model_type == "xgboost":
        dmat = xgb.DMatrix(dat)
        y_hat = model.predict(dmat)
    if model_type == "analytic":
        y_hat = model(dat)
    # Sum the y_hat deltas for each deviation for each feature:
    out = np.zeros(n_cols)
    row_ind = steps 
    for s in range(steps):
        base_y_hat = y_hat[s]
        for col_ind in range(n_cols):
            dev_y_hat = y_hat[row_ind]
            out[col_ind] += dev_y_hat - base_y_hat
            row_ind += 1
    return out


