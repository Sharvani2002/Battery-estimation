Xtrain_un = xlsread('data_set.xlsx',1,'B22:D25')';%reading input data from train excel sheet
Ytrain_un = xlsread('data_set.xlsx',1,'E22:E25')';%reading output data from train excel sheet
Xtrain = zeros(3,1,4);
Ytrain = zeros(1,1,4);

%{
for i=1:1:1
    Xtemp = Xtrain(:,(4*(i-1)+1):(4*i));
    Ytemp = Ytrain(:,(4*(i-1)+1):(4*i));
    Xtrain(:,i,:) = Xtemp;
    Ytrain(:,i,:) = Ytemp;
end
%}

Tx = 4;
%Tx time steps (so Xtrain_un and Ytrain_un  should have no. of rows multiple of Tx )
for i=1:1:1
    Xtemp = Xtrain(:,(Tx*(i-1)+1):(Tx*i));
    Ytemp = Ytrain(:,(Tx*(i-1)+1):(Tx*i));
    Xtrain(:,i,:) = Xtemp;
    Ytrain(:,i,:) = Ytemp;
end

Tx = 4; %=Ty
a0 = da0;
a0 = zeros([1, 1, Tx]);
a0(1,1,1) = 0.5;
% obtain parameters from trained data

[a, y_pred, caches] = rnn_forward(Xtrain, a0(1,1,1), parameters_Wax,parameters_Waa, parameters_Wya, parameters_ba, parameters_by);
loss = (1/2)*sum(abs(Ytrain-y_pred).^2,2);
fprintf('\nLoss in Test data: %f\n',loss);


