%{
Overview of the model:

Our model will have the following structure:
    Initialize parameters
    Run the optimization loop
    Forward propagation to compute the loss function
    Backward propagation to compute the gradients with respect to the loss function
    Clip the gradients to avoid exploding gradients
    Using the gradients, update our parameter with the gradient descent update rule.
[doesnot return now]Return the learned parameters
%}

%Edit this to add X and Y train data
Xtrain_un = xlsread('data_set.xlsx',1,'B2:D21')';%reading input data from train excel sheet
Ytrain_un = xlsread('data_set.xlsx',1,'E2:E21')';%reading output data from train excel sheet
Xtrain = zeros(3,5,4);
Ytrain = zeros(1,5,4);

%{
for i=1:1:5
    Xtemp = Xtrain_un(:,(4*(i-1)+1):(4*i));
    Ytemp = Ytrain_un(:,(4*(i-1)+1):(4*i));
    Xtrain(:,i,:) = Xtemp;
    Ytrain(:,i,:) = Ytemp;

end
%}

Tx = 4;
%Tx time steps (so Xtrain_un and Ytrain_un  should have no. of rows multiple of Tx )
for i=1:1:5
    Xtemp = Xtrain_un(:,(Tx*(i-1)+1):(Tx*i));
    Ytemp = Ytrain_un(:,(Tx*(i-1)+1):(Tx*i));
    Xtrain(:,i,:) = Xtemp;
    Ytrain(:,i,:) = Ytemp;

end
%Partition the training and test data. Train on the first 90% of the sequence and test on the last 10%.
%{
Xtrain_un = rand([4,5,4]);
Ytrain_un = rand([1,5,4]);

Xtrain = rand([4,2,4]);
Ytrain = rand([1,2,4]);
%}


%Normalize 
%{
min_value_X = min(Xtrain_un,2);
max_value_X = max(Xtrain_un,2);
Xtrain = (Xtrain - min_value_X)/(max_value_X - min_value_X);

min_value_Y = min(Ytrain_un,2);
max_value_Y = max(Ytrain_un,2);
Ytrain = (Ytrain - min_value_Y)/(max_value_Y - min_value_Y);
%}
%Standardize data
%{
mu_X = mean(Xtrain_un,2);
sig_X = std(Xtrain_un,2);
Xtrain = (Xtrain - mu_X) / sig_X;
mu_Y = mean(Ytrain_un,2);
sig_Y = std(Ytrain_un,2);
Ytrain = (Ytrain - mu_Y) / sig_Y;
%}
%{
Note the data is not being randomly shuffled before splitting. This is for two reasons:

It ensures that chopping the data into windows of consecutive samples is still possible.
It ensures that the validation/test results are more realistic, being evaluated on the data collected after the model was trained.
%}



%{
For adam optimizer, currently not used
epsilon = 1e-8;
beta1 = 0.9;
beta2 = 0.999;
learning_rate = 0.01;
num_iterations = 50;
t = 2;
%}

b = 0;
num_iterations = 200;
learning_rate = 0.0001;

[n_x,m,T_x] = size(Xtrain);
[n_y,m,T_y] = size(Ytrain);
n_a = 1;
a_prev = zeros([n_a,m]);

parameters_Wax = rand([n_a,n_x])*0.01;
parameters_Waa = rand([n_a,n_a])*0.01;
parameters_Wya = rand([n_y,n_a])*0.01;
parameters_ba = rand([n_a,1])*0.01;
parameters_by = rand([n_y,1])*0.01;

loss = 0;

for j = 1:1:num_iterations

    [loss_per_iter, dx, da0, dWax, dWaa, dba, parameters] = optimize(Xtrain, Ytrain, a_prev, parameters_Wax,parameters_Waa, parameters_Wya, parameters_ba, parameters_by, learning_rate);
    %Gradients are:
    %dx, da0, dWax, dWaa, dba, aa
    
    %get and update parameters
    parameters_Wax = parameters{1};
    parameters_Waa = parameters{2};
    parameters_Wya = parameters{3};
    parameters_ba = parameters{4};
    parameters_by = parameters{5};
    
    %loss = loss + loss_per_record;
     if mod(j,30) == 0
        fprintf('Iteration: %d, Loss: %f\n',j, loss_per_iter);
     end
    
end

    
