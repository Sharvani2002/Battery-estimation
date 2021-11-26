x_data = readmatrix('data from SOC estimation.xlsx','Sheet','Sheet1','Range','B2:D20001');
y_data = readmatrix('data from SOC estimation.xlsx','Sheet','Sheet1','Range','E2:E20001');
len = size(y_data,1)
n = size(x_data,2)

%normalising the data
[x_data,maxs,mins] = normalize(x_data,n);

%y_data = normalize(y_data,1)

x0 = [ones(len,1),x_data];

% gradient descent
repeat = 3500;
lrate = 0.3;
thetas = zeros(n+1, 1)
[best, costs] = gradient_descent(repeat, lrate, thetas, x0, y_data,len,n);

% plot costs
plot(1:repeat,costs);
