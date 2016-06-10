function theta = initializeParameters(hiddenSize, visibleSize)

%% Initialize parameters randomly based on layer sizes.
r  = sqrt(6) / sqrt(hiddenSize+visibleSize+1);   % we'll choose weights uniformly from the interval [-r, r] ,��Χ�����ؽڵ���Ŀ��Ȼ����صġ�
W1 = rand(hiddenSize, visibleSize) * 2 * r - r; %25*64 ��һ��Ĳ�����1�б�ʾ��һ�������ڵ��Ӧ�Ĳ�����1�б�ʾһ�����ؽڵ���������
W2 = rand(visibleSize, hiddenSize) * 2 * r - r; %64*25 �ڶ���Ĳ����� 1�б�ʾ��һ�����ز�ڵ�Ķ�Ӧ������1�б�ʾһ�������ڵ���������

b1 = zeros(hiddenSize, 1);% 25*1  ���ز�Ĳ���b����ʾ
b2 = zeros(visibleSize, 1);%64*1 �����Ĳ���b

% Convert weights and bias gradients to the vector form.
% This step will "unroll" (flatten and concatenate together) all 
% your parameters into a vector, which can then be used with minFunc. 
theta = [W1(:) ; W2(:) ; b1(:) ; b2(:)];

end

