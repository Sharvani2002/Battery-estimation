function [dxt, da_prev, dWax, dWaa, dba, dWay] = rnn_cell_backward(dy_next, da_next, cache)
    %{
    Implements the backward pass for the RNN-cell (single time-step).
    Arguments:
    da_next -- Gradient of loss with respect to next hidden state
    cache -- python dictionary containing useful values (output of rnn_cell_forward())

    Returns:
    gradients -- set of matrices containing:
                        dx -- Gradients of input data, of shape (n_x, m)
                        da_prev -- Gradients of previous hidden state, of shape (n_a, m)
                        dWax -- Gradients of input-to-hidden weights, of shape (n_a, n_x)
                        dWaa -- Gradients of hidden-to-hidden weights, of shape (n_a, n_a)
                        dba -- Gradients of bias vector, of shape (n_a, 1)
    %}
    
     % Retrieve values from cache
    %a_next, a_prev, xt, Wax, Waa, Wya, ba, by = cache;
    a_next = cache{1}; %a<t>
    a_prev = cache{2}; %a<t-1>
    xt = cache{3};
    Wax = cache{4};
    Waa = cache{5};
    Wya = cache{6};
    ba = cache{7};
    by = cache{8};
   
    % compute the gradient of tanh with respect to a_next
    dtanh = (1 - a_next.^2) .* da_next;

    % compute the gradient of the loss with respect to Wax 
    dxt = Wax' * dtanh; %dot product
    dWax = dtanh * xt'; %dot product

    % compute the gradient with respect to Waa
    da_prev = Waa' * dtanh; %dot product
    dWaa =  dtanh * a_prev'; %dot product

    % compute the gradient with respect to b 
    dba = sum(dtanh, 2);
    
    dWay =  a_next * dy_next';
    
end