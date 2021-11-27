function [a_next, yt_pred, cache] = rnn_cell_forward(xt, a_prev, parameters_Wax,parameters_Waa, parameters_Wya, parameters_ba, parameters_by)
    %{
    Implements a single forward step of the RNN-cell as described in
    Figure_RNN_cell

    Arguments:
    xt -- the input data at timestep "t", matrix of shape (n_x, m).
    a_prev -- Hidden state at timestep "t-1", matrix of shape (n_a, m)
    parameters -- matlab set of matrices containing:
                        Wax -- Weight matrix multiplying the input, matrix of shape (n_a, n_x)
                        Waa -- Weight matrix multiplying the hidden state, matrix of shape (n_a, n_a)
                        Wya -- Weight matrix relating the hidden-state to the output, matrix of shape (n_y, n_a)
                        ba --  Bias, matrix of shape (n_a, 1)
                        by -- Bias relating the hidden-state to the output, matrix of shape (n_y, 1)
    Returns:
    a_next -- next hidden state, of shape (n_a, m)
    yt_pred -- prediction at timestep "t", matrix of shape (n_y, m)
    cache -- tuple of values needed for the backward pass, contains (a_next, a_prev, xt, parameters)
    %}
    
    %Retrieve parameters from "parameters"
    Wax = parameters_Wax;
    Waa = parameters_Waa;
    Wya = parameters_Wya;
    ba = parameters_ba;
    by = parameters_by;

    %compute next activation state using the formula given above
    a_next = tanh(Waa * a_prev + Wax * xt + ba);
    %compute output of the current cell using the formula given above
    %yt_pred = softmax( Wya * a_next + by);
    yt_pred = Wya * a_next + by;
    
    %store values you need for backward propagation in cache
    cache = {a_next, a_prev, xt, parameters_Wax,parameters_Waa, parameters_Wya, parameters_ba, parameters_by};
    
end