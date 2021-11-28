function [a, y_pred, caches] = rnn_forward(x, a0, parameters_Wax,parameters_Waa, parameters_Wya, parameters_ba, parameters_by)
    %{
    Implementing the forward propagation of the recurrent neural network.

    Arguments:
    x -- Input data for every time-step, of shape (n_x, m, T_x).
    a0 -- Initial hidden state, of shape (n_a, m)
    parameters -- set of matrices containing:
                        Waa -- Weight matrix multiplying the hidden state, matrix of shape (n_a, n_a)
                        Wax -- Weight matrix multiplying the input, matrix of shape (n_a, n_x)
                        Wya -- Weight matrix relating the hidden-state to the output, matrix of shape (n_y, n_a)
                        ba --  Bias matrix of shape (n_a, 1)
                        by -- Bias relating the hidden-state to the output, matrix of shape (n_y, 1)

    Returns:
    a -- Hidden states for every time-step, matrix of shape (n_a, m, T_x)
    y_pred -- Predictions for every time-step, matrix of shape (n_y, m, T_x)
    caches -- tuple of values needed for the backward pass, contains (list of caches, x)
    %}
    
    % Initialize "caches" which will contain the list of all caches
    caches = {};
    
    % Retrieve dimensions from shapes of x and parameters["Wya"]
    %[n_x, m, T_x] = size(x);
    [n_x, m, T_x] = size(x);
    [n_y, n_a] = size(parameters_Wya);
    
    % initialize "a" and "y_pred" with zeros
    %a = zeros([n_a, m, T_x]);
    a = zeros([n_a, m, T_x]);
    %y_pred = zeros([n_y, m, T_x]);
    y_pred = zeros([n_y, m, T_x]);
    
    % Initialize a_next
    a_next = a0;
    
    % loop over all time-steps
    for t=1:1:(T_x)
        % Update next hidden state, compute the prediction, get the cache
        [a_next, yt_pred, cache] = rnn_cell_forward(x(:,:,t), a_next, parameters_Wax,parameters_Waa, parameters_Wya, parameters_ba, parameters_by);
        % Save the value of the new "next" hidden state in a
        a(:,:, t) = a_next;
        % Save the value of the prediction in y
        y_pred(:,:,t) = yt_pred;
        % Append "cache" to "caches"
        caches{end+1} = cache;
     end
    % store values needed for backward propagation in cache
    caches = {caches, x};
    
    
end