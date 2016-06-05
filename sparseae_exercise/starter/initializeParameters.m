function theta = initializeParameters(hiddenSize, visibleSize)

%% Initialize parameters randomly based on layer sizes.
r  = sqrt(6) / sqrt(hiddenSize+visibleSize+1);   % we'll choose weights uniformly from the interval [-r, r] ,��Χ�����ؽڵ���Ŀ��Ȼ����صġ�
W1 = rand(hiddenSize, visibleSize) * 2 * r - r; %25*64 ��ǰ��Ĳ���
W2 = rand(visibleSize, hiddenSize) * 2 * r - r; %64*25�Ƿ�������

b1 = zeros(hiddenSize, 1);% 25*1  ���ز�Ĳ���b
b2 = zeros(visibleSize, 1);%64*1 �����Ĳ���b

% Convert weights and bias gradients to the vector form.
% This step will "unroll" (flatten and concatenate together) all 
% your parameters into a vector, which can then be used with minFunc. 
theta = [W1(:) ; W2(:) ; b1(:) ; b2(:)];

end

