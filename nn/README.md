##Two-layer neural network

#Problem 14
Implement a neural network with one hidden layer. We suggest that you implement it in a way that it works with different kinds of optimisation algorithms. Use stochastic gradient descent with mini-batches and rmsprop as the initial optimisation method. Implement early stopping. You may follow the tutorial or, better, write your own implementation from ground up.<br />
Code: **problem_14.py**<br />
Result: **problem_14.odt**<br />
<br />

#Bonus question 2
Implement different regularisation approaches, like momentum, weight decay, l_1 regularisation, dropout.<br />
Code: **bonus_2.py**<br />
Result: **bonus_2** directory with file name **error [type of activation function] dropout [l10000] [l20000]**<br />
<br />

1. dropout indicates using dropout<br />
2. l1_ indicates using L1 regularization l1_p0001 -> value of L1 regularization is 0.0001<br />
3. l2_ indicates using L2 regularization l1_0 -> value of L2 regularization is 0<br />
<br />
**Depending on the parameter setting, the results are different.**<br />
<br />

In order to set the type of activation please change parameter of the function call<br />
"""<br />
sigmoid: sigmoid activation function<br />
tanh: tangent activation function<br />
relu: ReLU activation function<br />
"""<br />
**test_mlp([Type of activation function])*<br />


#Problem 15
Evaluate your implementation on MNIST. Initially use 300 hidden units with tanh activation functions.<br />
Code: **problem_15.py**<br />
Result: **problem_15_plots** directory with file name **error [type of activation function] [number of hidden units]**<br />
<br />
In order to set the type of activation please change parameter of the function call<br />
"""<br />
sigmoid: sigmoid activation function<br />
tanh: tangent activation function<br />
relu: ReLU activation function<br />
"""<br />
**test_mlp([Type of activation function])*<br />

#Problem 16
Try different nonlinear activation functions for the hidden units. Evaluate logistic sigmoid, tanh and rectified linear neurons in the hidden layer. Think about how the different activation functions look like and how they behave. Does—and if it does, how does—this influence, e.g., weight initialization or data preprocessing? Implement and test your reasoning in your code to see if the results support your conclusions.<br />
Code: **problem_16.py**<br />
Result: **problem_16_plots**<br />
<br />
In order to set the type of activation please change parameter of the function call<br />
"""<br />
sigmoid: sigmoid activation function<br />
tanh: tangent activation function<br />
relu: ReLU activation function<br />
"""<br />
**test_mlp([Type of activation function])*<br />

#Problem 17
Plot the error curves for the training, evaluation and test set for each of the activation functions evaluated in the previous problem into file error.png. That is, either provide one file with three subplots (one per activation function) and three error curves each, or provide three different files (error_
tanh.png, error_
sigmoid.png, and error_relu.png).<br />
Code: **problem_17.py**<br />
Result: **problem_17_plots**<br />
<br />
In order to set the type of activation please change parameter of the function call<br />
"""<br />
sigmoid: sigmoid activation function<br />
tanh: tangent activation function<br />
relu: ReLU activation function<br />
"""<br />
**test_mlp([Type of activation function])*<br />

#Problem 18
Visualize the receptive fields of the hidden layer and write them to file repflds.png. As in the previous problem, either provide one file with three subplots, or three distinct files.<br />
Code: **problem_18.py**<br />
Result: **problem_18_plots**<br />
<br />
In order to set the type of activation please change parameter of the function call<br />
"""<br />
sigmoid: sigmoid activation function<br />
tanh: tangent activation function<br />
relu: ReLU activation function<br />
"""<br />
**test_mlp([Type of activation function])*<br />

#Problem 19
Fine tune your implementation until you achieve an error rate of about 2%. Optionally try augmenting the training set as described in section 2. Do not spend too much time on this problem.<br />
Code: **problem_19.py**<br />
Result: **problem_19_plots**<br />
<br />
In order to set the type of activation please change parameter of the function call<br />
"""<br />
sigmoid: sigmoid activation function<br />
tanh: tangent activation function<br />
relu: ReLU activation function<br />
"""<br />
**test_mlp([Type of activation function])*<br />

#Theoretic prerequisites
artificial neural networks, backpropagation, optimization (gradient de-scent, rmsprop), receptive field

#Literature
Bishop: sections 5.1–5.2.1, 5.3.1<br />
Murphy: sections 16.5, 16.5.4, 16.5.6<br />
COURSERA Neural Networks for Machine Learning: lectures 6a, 6b, 6e<br />

#Tutorial
http://deeplearning.net/tutorial/mlp.html

#Reference
MNIST data set: http://deeplearning.net/data/mnist/mnist.pkl.gz<br />
................http://deeplearning.net/tutorial/gettingstarted.html<br />
CIFAR-10 data set: http://www.cs.toronto.edu/~kriz/cifar.html<br />
