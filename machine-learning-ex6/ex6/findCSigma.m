load('ex6data3.mat');

C = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
sigma = C;

m = size(C, 2);
n = size(sigma, 2);

J_vals = zeros(m, n);

for i = 1:m
	for j = 1:n
		
		model= svmTrain(X, y, C(i), @(x1, x2) gaussianKernel(x1, x2, sigma(j))); 
		predictions = svmPredict(model, Xval);
		J_vals(i, j) = mean(double(predictions ~= yval))
		
	endfor
endfor
size(J_vals)
J_vals = J_vals';
surf(C, sigma, J_vals);
