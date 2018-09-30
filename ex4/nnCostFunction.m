function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%



% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% add bias unit
X = [ones(size(X,1),1),X];

for i = 1:m
    
    %take one observations of the input layer [1 x input layer units]
    
    a1 = X(i,:);
    
    %multiply by mapping (i.e. weights) from input to hidden layer
   
   
    z2 = a1*Theta1';
    
    %sigmoid function
    
    a2 = sigmoid(z2);
    
    %add bias unit to hidden layer
    
    a2 = [ones(size(a2,1),1),a2];
    
    %%multiply by mapping (i.e. weights) from hidden to output layer
    
    z3 = a2*Theta2';
    
    %output nodes
    
    a3 = sigmoid(z3);
    
    %calculate cost by comparing to true result
    
    yVec = yVectorize(max(y),y(i)); %vectorize
    J = J + (log(a3)*yVec)+(log(1-a3)*(1-yVec));%purpose is that if Y is 1 and prediction is 0 penalty should be big and vice versa
    
    % calculate error at output nodes
    
    delta3 = a3' - yVec;
    
    % sum up delta
    
    Theta2_grad = Theta2_grad + delta3*a2 ;
    
    % calculate error at hidden layer
     
    delta2 = (delta3'*Theta2(:,2:end)).*sigmoidGradient(z2);
    
    % sum up delta
    Theta1_grad = Theta1_grad + delta2'*a1 ;
    
end


%regularization
if lambda > 0
    
    %intermediate
    Theta1_noBias = Theta1(:,2:end).^2;
    Theta2_noBias = Theta2(:,2:end).^2;
    penalty = (lambda/(2*m))*(sum(Theta1_noBias(:))+sum(Theta2_noBias(:)));
    
    %regularized result
    J = (-J/m) + penalty;
    Theta2_grad = Theta2_grad/m;
    Theta2_grad(:,2:end) = Theta2_grad(:,2:end) + (lambda/m)*Theta2(:,2:end);
    Theta1_grad = Theta1_grad/m;
    Theta1_grad(:,2:end) = Theta1_grad(:,2:end) + (lambda/m)*Theta1(:,2:end);
    
else
    %unregularized result
    J = (-J/m);
    Theta2_grad = Theta2_grad/m;
    Theta1_grad = Theta1_grad/m;

end


% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
