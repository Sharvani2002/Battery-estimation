function y = softmax(x)
    e_x = exp(x - max(x));
    y = e_x / sum(e_x, 1); %sum across axis = 0
end