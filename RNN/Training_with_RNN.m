%{
Overview of the model:

Our model will have the following structure:
    Initialize parameters
    Run the optimization loop
    Forward propagation to compute the loss function
    Backward propagation to compute the gradients with respect to the loss function
    [Not done]Clip the gradients to avoid exploding gradients
    Using the gradients, update our parameter with the gradient descent update rule.
[doesnot return now]Return the learned parameters
%}

%Edit this to padd X and Y train data
%{
Xtrain = xlsread('trainx.xlsx',1,'A1:C17000')';%reading input data from train excel sheet
Ytrain = xlsread('trainx.xlsx',1,'D1:D17000')';%reading output data from train excel sheet
%}

Xtrain = rand([20,4])';
Ytrain = rand([20,1])';

%{
For adam optimizer, currently not used
epsilon = 1e-8;
beta1 = 0.9;
beta2 = 0.999;
learning_rate = 0.01;
num_iterations = 50;
t = 2;
%}

%very simple a
% (1 * 1 dimension)
a_prev = 0;
b = 0;
num_iterations = 20;
learning_rate = 0.1;

parameters_Wax = rand([size(a_prev,1),size(Xtrain, 1)])*0.01;
parameters_Waa = rand([size(a_prev,1),size(a_prev, 2)])'*0.01;
parameters_Wya = rand([size(Ytrain,1),size(a_prev, 1)])*0.01;
parameters_ba = rand([size(b,1),size(a_prev, 1)])*0.01;
parameters_by = rand([size(b,1),size(Ytrain, 1)])*0.01;


for j = 1:1:num_iterations
    %v_dW,v_db, s_dW, s_db = initialize_adam(parameters_W,parameters_b);
    %curr_loss, gradients, a_prev = optimize(X, Y, a_prev, parameters)
    %parameters_W,parameters_b,v_dW,v_db, s_dW, s_db = update_parameters_with_adam(parameters_W,parameters_b, grads_dW,grads_db, v_dW,v_db, s_dW,s_db, t, learning_rate, beta1, beta2,  epsilon);
    
    [loss, dx, da0, dWax, dWaa, dba, aa, parameters] = optimize(Xtrain, Ytrain, a_prev, parameters_Wax,parameters_Waa, parameters_Wya, parameters_ba, parameters_by, learning_rate);
    %Gradients are:
    %dx, da0, dWax, dWaa, dba, aa
    
    %get and update parameters
    parameters_Wax = parameters{1};
    parameters_Waa = parameters{2};
    parameters_Wya = parameters{3};
    parameters_ba = parameters{4};
    parameters_by = parameters{5};

     if mod(j,10) == 0
        fprintf('Iteration: %d, Loss: %f\n',j, loss);
     end
    
end

    
