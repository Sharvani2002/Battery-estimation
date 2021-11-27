function [loss, dx, da0, dWax, dWaa, dba, aa, parameters] = optimize(X, Y, a_prev, parameters_Wax,parameters_Waa, parameters_Wya, parameters_ba, parameters_by, learning_rate)

% learning_rate = 0.01
%{
    Execute one step of the optimization to train the model.
    
    Arguments:
    X -- list of integers, where each integer is a number that maps to a character in the vocabulary.
    Y -- list of integers, exactly the same as X but shifted one index to the left.
    a_prev -- previous hidden state.
    parameters -- python dictionary containing:
                        Wax -- Weight matrix multiplying the input, numpy array of shape (n_a, n_x)
                        Waa -- Weight matrix multiplying the hidden state, numpy array of shape (n_a, n_a)
                        Wya -- Weight matrix relating the hidden-state to the output, numpy array of shape (n_y, n_a)
                        b --  Bias, numpy array of shape (n_a, 1)
                        by -- Bias relating the hidden-state to the output, numpy array of shape (n_y, 1)
    learning_rate -- learning rate for the model.
    
    Returns:
    loss -- value of the loss function (cross-entropy)
    gradients -- python dictionary containing:
                        dWax -- Gradients of input-to-hidden weights, of shape (n_a, n_x)
                        dWaa -- Gradients of hidden-to-hidden weights, of shape (n_a, n_a)
                        dWya -- Gradients of hidden-to-output weights, of shape (n_y, n_a)
                        db -- Gradients of bias vector, of shape (n_a, 1)
                        dby -- Gradients of output bias vector, of shape (n_y, 1)
    a[len(X)-1] -- the last hidden state, of shape (n_a, 1)
%}
    
    % Forward propagate through time
    [a, y_pred, caches] = rnn_forward(X, a_prev,  parameters_Wax,parameters_Waa, parameters_Wya, parameters_ba, parameters_by);
    
    %Find loss
    loss = sum(abs(Y-y_pred).^2);
    
    % Backpropagate through time 
    [dx, da0, dWax, dWaa, dba, a] = rnn_backward(a, caches);
    
    % Clip your gradients between -5 (min) and 5 (max)
    %gradients = clip(gradients, 5)

    %Update parameters
    parameters_Wax = parameters_Wax -learning_rate * dWax;
    parameters_Waa = parameters_Waa -learning_rate * dWaa;
    parameters_Wya = parameters_Wya -learning_rate * dWya;
    parameters_ba  = parameters_ba -learning_rate * db;
    parameters_by = parameters_by -learning_rate * dby;
    
    parameters = {parameters_Wax,parameters_Waa,parameters_Wya,parameters_ba,parameters_by};
    aa = a(length(X)-1);

end