function [loss, dx, da0, dWax, dWaa, dba, parameters] = optimize(X, Y, a_prev, parameters_Wax,parameters_Waa, parameters_Wya, parameters_ba, parameters_by, learning_rate)

%{
  Execute one step of the optimization to train the model.
    
    Arguments:
    X -- input data
    Y -- output data
    a_prev -- previous hidden state.
    parameters -- matlab array containing:
                        Wax -- Weight matrix multiplying the input, matrix of shape (n_a, n_x)
                        Waa -- Weight matrix multiplying the hidden state, matrix of shape (n_a, n_a)
                        Wya -- Weight matrix relating the hidden-state to the output, matrix of shape (n_y, n_a)
                        b --  Bias, matrix of shape (n_a, 1)
                        by -- Bias relating the hidden-state to the output, matrix of shape (n_y, 1)
    learning_rate -- learning rate for the model.
    
    Returns:
    loss -- value of the loss function (cross-entropy)
    gradients -- set of matlab matrices containing:
                        dWax -- Gradients of input-to-hidden weights, of shape (n_a, n_x)
                        dWaa -- Gradients of hidden-to-hidden weights, of shape (n_a, n_a)
                        dWya -- Gradients of hidden-to-output weights, of shape (n_y, n_a)
                        db -- Gradients of bias vector, of shape (n_a, 1)
                        dby -- Gradients of output bias vector, of shape (n_y, 1)
%}
    
    % Forward propagate through time
    [a, y_pred, caches] = rnn_forward(X, a_prev,  parameters_Wax,parameters_Waa, parameters_Wya, parameters_ba, parameters_by);
    
    
    %Find loss
    loss = (1/2)*sum(abs(Y-y_pred).^2,2);
    dy = (Y-y_pred); %(ny, m, Ty) dimension
    W_ay = parameters_Wya';
    da = W_ay * dy; %(na, m, Ty) dimension
    m = size(dy,2);
    dby = sum(sum(dy,2)/m, 3); %(n_y,1) dimension
    % Backpropagate through time 
    [dx, da0, dWax, dWaa, dba, a, dWay] = rnn_backward(dy, da, caches);
    
    % Clip the gradients between -5 (min) and 5 (max)
    max_clip_value = 5;
    min_clip_value = -5;
    dx = clip(dx, max_clip_value, min_clip_value);
    da0 = clip(da0, max_clip_value, min_clip_value);
    dWax = clip(dWax, max_clip_value, min_clip_value);
    dWaa = clip(dWaa, max_clip_value, min_clip_value);
    dba = clip(dba, max_clip_value, min_clip_value);
    dWay = clip(dWay, max_clip_value, min_clip_value);
    dby = clip(dby, max_clip_value, min_clip_value);
                

    %Update parameters
    parameters_Wax = parameters_Wax -learning_rate * dWax;
    parameters_Waa = parameters_Waa -learning_rate * dWaa;
    parameters_Wya = parameters_Wya -learning_rate * (dWay)';
    parameters_ba  = parameters_ba -learning_rate * dba;
    parameters_by = parameters_by -learning_rate * dby;
    
    parameters = {parameters_Wax,parameters_Waa,parameters_Wya,parameters_ba,parameters_by};

end