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

%data: 64*1000
W1 = reshape(theta(1:hiddenSize*visibleSize), hiddenSize, visibleSize);%25*64 
W2 = reshape(theta(hiddenSize*visibleSize+1:2*hiddenSize*visibleSize), visibleSize, hiddenSize);%64*25
b1 = theta(2*hiddenSize*visibleSize+1:2*hiddenSize*visibleSize+hiddenSize);%25*1
b2 = theta(2*hiddenSize*visibleSize+hiddenSize+1:end);%62*1

% Cost and gradient variables (your code needs to compute these values). 
% Here, we initialize them to zeros. 
cost = 0;
W1grad = zeros(size(W1)); %25*64
W2grad = zeros(size(W2));%64*26
b1grad = zeros(size(b1)); %25*1
b2grad = zeros(size(b2));%64*1

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

%隐藏层的节点输出
% m=10;
% 
% for i=1:m:size(data,2)
% 
%     deltaW_up_2=zeros(size(W2));
%     deltaB_up_2=zeros(size(b2));
%     
%     deltaW_up_1=zeros(size(W1));
%     deltaB_up_1=zeros(size(b1));
%     for j=i:1:i+m-1
%         %样本，单个进行训练
%         x=data(:,j);
%         a_up_2=W1*x;
%         z_up_2=sigmoid(a_up_2); %25*1000
%         a_up_3=W2*z_up_2;
%         z_up_3=sigmoid(a_up_3);%64*1000
%         error=z_up_3-x;%64*1000 误差
%         sita_up_3=error.*sigmoid_derivative(z_up_3);%输出层的值,sita调整
%         sita_up_2=W2'*sita_up_3.*sigmoid_derivative(z_up_2); %隐藏层的sita调整
%         %
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
%     end
%     
%     W2=W2-alpha*((1/m*deltaW_up_2)+lambda*W2);
%     b2=b2-alpha*(1/m*deltaB_up_2);
%     
%     W1=W1-alpha*((1/m*deltaW_up_1)+lambda*W1);
%     b1=b1-alpha*(1/m*deltaB_up_1);
%     
% end


    deltaW_up_2=zeros(size(W2));
    deltaB_up_2=zeros(size(b2));
    
    deltaW_up_1=zeros(size(W1));
    deltaB_up_1=zeros(size(b1));
    m=size(data,2);
    
    rho_temp=sigmoid(data'*W1');
    rho_hat=mean(rho_temp)';
    
for i=1:1:size(data,2)

        %样本，单个进行训练
        x=data(:,i);
        z_up_2=W1*x;
        a_up_2=sigmoid(z_up_2); %25*1000
        z_up_3=W2*a_up_2;
        a_up_3=sigmoid(z_up_3);%64*1000
        error=a_up_3-x;%64*1000 误差
        sita_up_3=error.*sigmoid_derivative(z_up_3);%输出层的值,sita调整
        sita_up_2=(  W2'*sita_up_3+ beta.*((-sparsityParam./rho_hat)+(1-sparsityParam)./(1-rho_hat))  ).*sigmoid_derivative(z_up_2); %隐藏层的sita调整 
        radaJ_div_radaW_up_2=sita_up_3*a_up_2';
        deltaW_up_2=deltaW_up_2+radaJ_div_radaW_up_2;
       
        radaJ_div_radaB_up_2=sita_up_3;
        deltaB_up_2=deltaB_up_2+radaJ_div_radaB_up_2;
        
        radaJ_div_radaW_up_1=sita_up_2*x';
        deltaW_up_1=deltaW_up_1+radaJ_div_radaW_up_1;
        
        radaJ_div_radaB_up_1=sita_up_2;
        deltaB_up_1=deltaB_up_1+radaJ_div_radaB_up_1;
   
end


   W2grad=(1/m*deltaW_up_2)+lambda*W2;
   b2grad=1/m*deltaB_up_2;
    
    W1grad=(1/m*deltaW_up_1)+lambda*W1;
    b1grad=1/m*deltaB_up_1;







%-------------------------------------------------------------------
% After computing the cost and gradient, we will convert the gradients back
% to a vector format (suitable for minFunc).  Specifically, we will unroll
% your gradient matrices into a vector.

grad = [W1grad(:) ; W2grad(:) ; b1grad(:) ; b2grad(:)];

end

%-------------------------------------------------------------------
% Here's an implementation of the sigmoid function, which you may find useful
% in your computation of the costs and the gradients.  This inputs a (row or
% column) vector (say (z1, z2, z3)) and returns (f(z1), f(z2), f(z3)). 


function a=test()
    % 检查sigmoid
    a=(sigmoid(0.5+0.0001)- sigmoid(0.5-0.0001))/0.0002
    b=sigmoid_derivative(0.5)
end

