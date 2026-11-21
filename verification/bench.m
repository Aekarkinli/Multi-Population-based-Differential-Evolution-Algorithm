function f = bench(X, mydata)
% Row-wise benchmark functions used for the MATLAB <-> Python check.
switch mydata.name
    case 'sphere'
        f = sum(X.^2, 2);
    case 'rastrigin'
        f = sum(X.^2 - 10*cos(2*pi*X) + 10, 2);
    case 'rosenbrock'
        f = sum(100*(X(:,2:end) - X(:,1:end-1).^2).^2 + (1 - X(:,1:end-1)).^2, 2);
    case 'ackley'
        D = size(X,2);
        f = -20*exp(-0.2*sqrt(sum(X.^2,2)/D)) - exp(sum(cos(2*pi*X),2)/D) + 20 + exp(1);
    case 'shifted_sphere'   % optimum at 3.0, box [1,5] -> both bounds positive
        f = sum((X-3).^2, 2);
end
