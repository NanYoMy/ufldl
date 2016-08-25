function sigm_deri = sigmoid_derivative(x)

    sigm_deri=sigmoid(x).*(1-sigmoid(x));
end