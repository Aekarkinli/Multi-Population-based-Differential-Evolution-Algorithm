function [out, x0, y0, r] = fitCircle(X, mydata)
% Objective function of the circle-fitting example (see example_circle_fit.m).
% X      : (n x 3) matrix, each row is a candidate [x0, y0, r]
% mydata : struct with fields x and y (point cloud coordinates)
% out    : (n x 1) vector, sum of absolute algebraic residuals
x = mydata.x;
y = mydata.y;
n = size(X, 1);
out = zeros(n, 1);
for i = 1:n
    [x0, y0, r] = deal(X(i,1), X(i,2), abs(X(i,3)));
    out(i) = sum(abs((x - x0).^2 + (y - y0).^2 - r^2));
end
