function [cost,grad] = sparseAutoencoderCost(theta, visibleSize, hiddenSize, ...
                                             lambda, sparsityParam, beta, data)

% visibleSize: the number of input units (probably 64) 
% hiddenSize: the number of hidden units (probably 25) 
% lambda: weight decay parameter
% sparsityParam: The desired average activation for the hidden units (denoted in the lecture
%                           notes by the greek alphabet rho, which looks like a lower-case "p").
% beta: weight of sparsity penalty term
% data: Our 64x10000 matrix containing the training data.  So, data(:,i) is the i-th training example. 
  
% The input theta is a vector (because minFunc expects the parameters to be a vector). 
% We first convert theta to the (W1, W2, b1, b2) matrix/vector format, so that this 
% follows the notation convention of the lecture notes. 

%data: 64*1000'
rho=sparsityParam;
W1 = reshape(theta(1:hiddenSize*visibleSize), hiddenSize, visibleSize);%25*64 
W2 = reshape(theta(hiddenSize*visibleSize+1:2*hiddenSize*visibleSize), visibleSize, hiddenSize);%64*25
b1 = theta(2*hiddenSize*visibleSize+1:2*hiddenSize*visibleSize+hiddenSize);%25*1
b2 = theta(2*hiddenSize*visibleSize+hiddenSize+1:end);%62*1

% Cost and gradient variables (your code needs to compute these values). 
% Here, we initialize them to zeros. 


%% ---------- YOUR CODE HERE --------------------------------------
%  Instructions: Compute the cost/optimization objective J_sparse(W,b) for the Sparse Autoencoder,
%                and the corresponding gradients W1grad, W2grad, b1grad, b2grad.
%
% W1grad, W2grad, b1grad and b2grad should be computed using backpropagation.
% Note that W1grad has the same dimensions as W1, b1grad has the same dimensions
% as b1, etc.  Your code should set W1grad to be the partial derivative of J_sparse(W,b) with
% respect to W1.  I.e., W1grad(i,j) should be the partial derivative of J_sparse(W,b) 
% with respect to the input parameter W1(i,j).  Thus, W1grad should be equal to the term 
% [(1/m) \Delta W^{(1)} + \lambda W^{(1)}] in the last block of pseudo-code in Section 2.2 
% of the lecture notes (and similarly for W2grad, b1grad, b2grad).
% 
% Stated differently, if we were using batch gradient descent to optimize the parameters,
% the gradient descent update to W1 would be W1 := W1 - alpha * W1grad, and similarly for W2, b1, b2. 
% 

%B=repmat( [1 2;3 4],2,3)
%B = 
%1      2      1     2    1    2
%3      4      3     4    3    4
%1     2     1     2     1     2
%3     4     3     4     3     4



[dim,sampleNum]=size(data)
cost = 0;
W1grad = zeros(size(W1)); %25*64
W2grad = zeros(size(W2));%64*26
b1grad = zeros(size(b1)); %25*1
b2grad = zeros(size(b2));%64*1

a1=data;
z1=data;
z2=W1*a1+repmat(b1,1,sampleNum);
a2=sigmoid(z2);
z3=W2*a2+repmat(b2,1,sampleNum);
a3=sigmoid(z3);
%auto sparse encoder的误差
rho_j=sum(a2,2)/sampleNum;
cost=1/sampleNum*0.5*sum(sum((data-a3).^2))+lambda/2*(sum(sum(W1.^2))+sum(sum(W2.^2)))+beta*sum(rho*log(rho./rho_j)+(1-rho)*log((1-rho)./(1-rho_j)));


deltaW_up_2=zeros(size(W2));
deltaB_up_2=zeros(size(b2));    
deltaW_up_1=zeros(size(W1));
deltaB_up_1=zeros(size(b1));
m=size(data,2);

sita3=(a3-data).*sigmoid_derivative(z3);
sita2=(W2'*sita3+ repmat(beta.*(((-sparsityParam)./rho_j)+(1-sparsityParam)./(1-rho_j)) ,1,m )).*sigmoid_derivative(z2);

deltaB_up_2=mean(sita3,2);
deltaB_up_1=mean(sita2,2);

%deltaW_up_2=sita3*a2';
%deltaW_up_1=sita2*a1';

for i = 1 : sampleNum
    deltaW_up_1 = deltaW_up_1 + sita2(:,i) * a1(:,i)'; % 25 * 10000 * 10000 * 64 = 25 * 64
    deltaW_up_2 = deltaW_up_2 + sita3(:,i) * a2(:,i)'; %  64 * 10000 * 10000 * 25 = 64 * 25
end

W2grad=(deltaW_up_2/sampleNum)+lambda*W2;
b2grad=deltaB_up_2;
W1grad=(deltaW_up_1/sampleNum)+lambda*W1;
b1grad=deltaB_up_1;


% for i=1:1:size(data,2)
%         %样本，单个进行训练
%         x=data(:,i);
%         z_up_2=W1*x+b1;
%         a_up_2=sigmoid(z_up_2); %25*1000
%         z_up_3=W2*a_up_2+b2;
%         a_up_3=sigmoid(z_up_3);%64*1000
%         error=a_up_3-x;%64*1000 误差
%         sita_up_3=error.*sigmoid_derivative(z_up_3);%输出层的值,sita调整
%         sita_up_2=(W2'*sita_up_3+ beta.*(((-sparsityParam)./rho_j)+(1-sparsityParam)./(1-rho_j))  ).*sigmoid_derivative(z_up_2); %隐藏层的sita调整 
%         radaJ_div_radaW_up_2=sita_up_3*a_up_2';
%         deltaW_up_2=deltaW_up_2+radaJ_div_radaW_up_2;
%        
%         radaJ_div_radaB_up_2=sita_up_3;
%         deltaB_up_2=deltaB_up_2+radaJ_div_radaB_up_2;
%         
%         radaJ_div_radaW_up_1=sita_up_2*x';
%         deltaW_up_1=deltaW_up_1+radaJ_div_radaW_up_1;
%         
%         radaJ_div_radaB_up_1=sita_up_2;
%         deltaB_up_1=deltaB_up_1+radaJ_div_radaB_up_1;
%    
% end
% W2grad=(1/m*deltaW_up_2)+lambda*W2;
% b2grad=1/m*deltaB_up_2;
% W1grad=(1/m*deltaW_up_1)+lambda*W1;
% b1grad=1/m*deltaB_up_1;



%%%
%这个是正确的版本，
%%%
% [dim, sampleNum] = size(data);
% cost = 0;
% 
% b1 = b1';
% b1 = b1(ones(1, sampleNum), :);
% b1 = b1';
% 
% z2 = W1 * data + b1;
% a2 = sigmoid(z2); % 25 * 10000
% 
% b2 = b2';
% b2 = b2(ones(1, sampleNum), :);
% b2 = b2';
% 
% z3 = W2 * a2 + b2;
% a3 = sigmoid(z3); % 64 * 10000
% cost = 0.5 / sampleNum * sum(sum((a3 - data).^2));
% cost = cost + 0.5 * lambda * (sum(sum(W1.^2)) + sum(sum(W2.^2)));
% rothj = sum(a2,2) / sampleNum;
% cost = cost + beta * sum(sparsityParam * log(sparsityParam ./ rothj) + ...
%     (1-sparsityParam) * log((1-sparsityParam) ./ (1-rothj)));
% delta3 = (a3 - data) .* sigmoid(z3) .* (1 - sigmoid(z3)); % 64 * 10000
% delta2 = (W2' * delta3 + repmat((beta * (-sparsityParam ./ rothj + (1-sparsityParam) ./ (1-rothj) ) ), 1, sampleNum))...
%     .* sigmoid(z2) .* (1 - sigmoid(z2)); % 25 * 64 * 64 * 10000 = 25 * 10000
% % f'(x) = f(x) * (1 - f(x))
% % 非向量化版本
% for i = 1 : sampleNum
%     W1grad = W1grad + delta2(:,i) * data(:,i)'; % 25 * 10000 * 10000 * 64 = 25 * 64
%     W2grad = W2grad + delta3(:,i) * a2(:,i)'; %  64 * 10000 * 10000 * 25 = 64 * 25
% end
% %  向量化版本
% 
% W1grad = W1grad / sampleNum;
% W2grad = W2grad / sampleNum;
% W1grad = W1grad + lambda * W1;
% W2grad = W2grad + lambda * W2;
% b1grad = sum(delta2') / sampleNum;
% b2grad = sum(delta3') / sampleNum;
% -------------------------------------------------------------------
% After computing the cost and gradient, we will convert the gradients back
% to a vector format (suitable for minFunc).  Specifically, we will unroll
% your gradient matrices into a vector.

grad = [W1grad(:) ; W2grad(:) ; b1grad(:) ; b2grad(:)];

end

%-------------------------------------------------------------------
% Here's an implementation of the sigmoid function, which you may find useful
% in your computation of the costs and the gradients.  This inputs a (row or
% column) vector (say (z1, z2, z3)) and returns (f(z1), f(z2), f(z3)). 




