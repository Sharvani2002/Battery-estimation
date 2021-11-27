function [v_dW,v_db, s_dW, s_db] = initialize_adam(parameters_W,parameters_b)
    %number of layers in the neural networks
    L = floor(length(parameters_W)); 
    %Initialize v, s. 
    v_dW = []; v_db = [];
    s_dW = []; s_db = [];
    %Input: "parameters". 
    %Outputs: "v, s".
    for l = 1:1:L
        v_dW(l) = zeros(size(parameters_W(l)));
        v_db(l)= zeros(size(parameters_b(l)));
        s_dW(l)= zeros(size(parameters_W(l)));
        s_db(l) = zeros(size(parameters_b(l)));
    end
end