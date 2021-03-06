function [parameters_W,parameters_b,v_dW,v_db, s_dW, s_db] = update_parameters_with_adam(parameters_W,parameters_b, grads_dW,grads_db, v_dW,v_db, s_dW,s_db, t, learning_rate, beta1, beta2,  epsilon)
%can keep these as default:
%{
epsilon = 1e-8;
beta1 = 0.9;
beta2 = 0.999;
learning_rate = 0.01;
%}

%{
    Update parameters using Adam
    
    Arguments:
    parameters_W,parameters_b -- matlab matrices containing parameters W,b
    grads_dW, grads_db -- matlab vectors containing the gradients for each parameters:
                    grads_dW1 = dW1
                    grads_db1 = db1
    v_dW,v_db -- Adam variables, moving average of the first gradient
    s_dW,s_db -- Adam variables, moving average of the squared gradient
    learning_rate -- the learning rate, scalar.
    beta1 -- Exponential decay hyperparameter for the first moment estimates 
    beta2 -- Exponential decay hyperparameter for the second moment estimates 
    epsilon -- hyperparameter preventing division by zero in Adam updates
    Returns:
    parameters_W,parameters_b -- matlab matrices containing the updated parameters 
    v_dW,v_db -- Adam variables, moving average of the first gradient
    s_dW, s_db -- Adam variables, moving average of the squared gradient
    %}
    
    L = floor(length(parameters_W));                %number of layers in the neural networks
    v_corrected_dW = []; v_corrected_db = [];       %Initializing first moment estimate, python dictionary
    s_corrected_dW = []; s_corrected_db = [];       %Initializing second moment estimate, python dictionary
    
    %Perform Adam update on all parameters
    for l=1:1:L
        %Moving average of the gradients. Inputs: "v, grads, beta1". Output: "v".
        v_dW(l)= beta1 * v_dW(l) + (1 - beta1) * grads_dW(l);
        v_db(l) = beta1 * v_db(l) + (1 - beta1) * grads_db(l);

        %Compute bias-corrected first moment estimate. Inputs: "v, beta1, t". Output: "v_corrected".
        v_corrected_dW(l) = v_dW(l)/ (1 - beta1^t);
        v_corrected_db(l) = v_db(l) / (1 - beta1^t);

        %Moving average of the squared gradients. Inputs: "s, grads, beta2". Output: "s".
        s_dW(l) = beta2 * s_dW(l) + (1 - beta2) * (grads_dW(l)^2);
        s_db(l) = beta2 * s_db(l) + (1 - beta2) * (grads_db(l)^2);

        % Compute bias-corrected second raw moment estimate. Inputs: "s, beta2, t". Output: "s_corrected".
        s_corrected_dW(l) = s_dW(l) / (1 - beta2^t);
        s_corrected_db(l) = s_db(l) / (1 - beta2^t);

        % Update parameters. Inputs: "parameters, learning_rate, v_corrected, s_corrected, epsilon". Output: "parameters".
        parameters_W(l) = parameters_W(l) - learning_rate * v_corrected_dW(l) / sqrt(s_corrected_dW(l) + epsilon);
        parameters_b(l) = parameters_b(l) - learning_rate * v_corrected_db(l) / sqrt(s_corrected_db(l) + epsilon);

    end
end