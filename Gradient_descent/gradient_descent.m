function [thetas, costs] = gradient_descent(repeat, lrate, thetas, x0, y_data, len, n)
 costs = zeros(repeat,1);
 for r = 1:repeat
   hc = x0 * thetas - y_data;
   temp = sum(hc .* x0);
   thetas = thetas - (lrate * (1/len)) * temp';
   costs(r) = cost(thetas, x0, y_data);
 end
end
