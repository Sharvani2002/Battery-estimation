function [dx, da0, dWax, dWaa, dba, a, dWay] = rnn_backward(dy,da, caches)
    %{
    Implement the backward pass for a RNN over an entire sequence of input data.

    Arguments:
    da -- Upstream gradients of all hidden states, of shape (n_a, m, T_x)
    caches -- tuple containing information from the forward pass (rnn_forward)
    
    Returns:
    gradients -- set of matrices containing:
                        dx -- Gradient w.r.t. the input data, matrix of shape (n_x, m, T_x)
                        da0 -- Gradient w.r.t the initial hidden state, matrix of shape (n_a, m)
                        dWax -- Gradient w.r.t the input's weight matrix, matrix of shape (n_a, n_x)
                        dWaa -- Gradient w.r.t the hidden state's weight matrix, matrixof shape (n_a, n_a)
                        dba -- Gradient w.r.t the bias, of shape (n_a, 1)
    %}

    % Retrieve values from the first cache (t=1) of caches
    
    x = caches{2};
    caches = caches{1};
    
    caches1 = caches{1};
    a1 = caches1{1};
    a0 = caches1{2};
    x1 = caches1{3};
    %parameters = caches_1{4:};
   
    % Retrieve dimensions from da's and x1's shapes
    %[n_a, m, T_x] = size(da);
    [n_a,m, T_x] = size(da);
    [n_x, m] = size(x1);
    [n_y, m, T_y] = size(dy);
    % Initialize the gradients with the right sizes
    dx = zeros([n_x, m, T_x]);
    dWax = zeros([n_a, n_x]);
    dWaa = zeros([n_a, n_a]);
    dba = zeros([n_a, 1]);
    da0 = zeros([n_a,m]);
    da_prevt = zeros([n_a,m]);
    dWay = zeros([n_a, n_y]);
  
    da_prev = da(:,:,T_x);
    % Loop through all the time steps
    for t = T_x:-1:1
        da(:,:,t) = da_prev;
        % Compute gradients at time step t. Choose wisely the "da_next" and the "cache" to use in the backward propagation step.
       %[dxt, da_prevt, dWaxt, dWaat, dbat, dWayt] = rnn_cell_backward(dy(:,:,t), da(:,:,t) + da_prevt, caches{t});
       [dxt, da_prevt, dWaxt, dWaat, dbat, dWayt] = rnn_cell_backward(dy(:,:,t), da(:,:,t), caches{t});
       
        % Increment global derivatives w.r.t parameters by adding their derivative at time-step t 
        dx(:,:, t) = dxt;
        da_prev = da_prevt;
        dWax = dWax + dWaxt;
        dWaa = dWaa + dWaat;
        dba = dba + dbat;
        dWay = dWay + dWayt;
        
    % Set da0 to the gradient of a which has been backpropagated through all time-steps 
    da0 = da_prevt;
    
    end
    
    a = a1;
    
end