function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%

h = zeros(m,1);

%for i = 1:m
	%sum += y(i)*(log10(sigmoid(X(i,:)*theta))) + (1 - y(i))*(1 - log10(sigmoid(X(i,:)*theta)));	
%	h(i) = sigmoid(X(i,:)*theta);
%endfor

h = sigmoid(X*theta);


J = -1*sum(y.*log(h) + (1-y).*log(1-h))/m;

%gSum = 0;
%for i = 1:m
%	gSum += (sigmoid(X(i,:)*theta) - y(i))*X(i,:);
%endfor

grad = (X'*(h - y))/m;

%gSum = gSum';

%grad = gSum/m;

% =============================================================

end
