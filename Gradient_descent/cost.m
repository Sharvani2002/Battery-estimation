function j = cost(thetas, x0, y_data)
  hc = x0 * thetas - y_data;
  m = length(y_data);
  j = (hc' * hc) / (2 * m);
end
