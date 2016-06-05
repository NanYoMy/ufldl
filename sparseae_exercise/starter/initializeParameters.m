function theta = initializeParameters(hiddenSize, visibleSize)

%% Initialize parameters randomly based on layer sizes.
r  = sqrt(6) / sqrt(hiddenSize+visibleSize+1);   % we'll choose weights uniformly from the interval [-r, r] ,范围和隐藏节点数目竟然是相关的。
W1 = rand(hiddenSize, visibleSize) * 2 * r - r; %25*64 是前向的参数
W2 = rand(visibleSize, hiddenSize) * 2 * r - r; %64*25是反向误差导数

b1 = zeros(hiddenSize, 1);% 25*1  隐藏层的参数b
b2 = zeros(visibleSize, 1);%64*1 输出层的参数b

% Convert weights and bias gradients to the vector form.
% This step will "unroll" (flatten and concatenate together) all 
% your parameters into a vector, which can then be used with minFunc. 
theta = [W1(:) ; W2(:) ; b1(:) ; b2(:)];

end

