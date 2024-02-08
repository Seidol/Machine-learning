# import matplotlib.pyplot as plt
# import numpy as np
#
# # X=np.random.randn(3,4)
# # X=np.array([2.1,2.2,-1.8])
# # W=[[[5,3.1,2],[2.3,-2.8,3.1]],[[3.1,2.5,3.3],[3.1,2.5,3.3]]]
# # W=[[5,3.1,2],[2.3,-2.8,3.1],[3.1,2.5,3.3],[3.1,2.5,3.3]]
# # W=[[5,3.1],[2.3,-2.8,3.1],[3.1,2.5,3.3],[3.1,2.5,3.3]]
# # W=np.array(W)
# # print(W.shape)
# # print(W.ndim)
# # print(type(X))
# # print(W)
# # print(X.ndim)
# # print(X.shape)
# # print(X)
# # print('Number of dimension for X = ',X.ndim)
#
# # Neuron network with learning
#
#
# input = [2.1, 2.3, 3.4]
# # bias=[3.3,4.4]
# # w1=[[5,3.1],[2.3,-2.8],[3.1,2.5]]
# # w1=np.array(w1)
# # print(w1.ndim)
# # print(w1.shape)
# #
# # y=np.dot(w1.T,input)+bias
# # print(y)
# # print('Traspose = ',w1.T)
# # print('bias',bias.shape)
#
# # Tenser(object) : the something looking
#
# X = np.random.randn(3, 4)
#
#
# # input=[2.1,2.3,3.4]
# class Dense_leyer:
#     def __init__(self, inputs, neurons):
#         self.w1 = 0.2 * np.random.randn(inputs, neurons)
#         self.bias = 0.3 * np.random.randn(1, neurons)
#
#     def forward(self, inputdata):
#         self.output = np.dot(inputdata, self.w1) + self.bias
#         return self.output
#
#
# Layer1 = Dense_leyer(4, 5)
# Layer1.forward(X)
#
# # Layer 2
# Layer2 = Dense_leyer(5, 2)
# Layer2.forward(Layer1.output)
#
# print('\n ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ \n')
# print('First  Dense Layer output = \n', Layer1.output)
# print('\n ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ \n')
# print(' 2nd is the neuron network output = \n', Layer2.output)
# print('\n ++++++++++ Neuron Network of linear end here +++++++++++++++++ \n')
#
# # display image.
# #
# # X=np.load('X.npy')
# # print('x250:\n ',X[250])
# # plt.imshow(X[250])
# # plt.show()
# # print('x250 shape:\n ',X[250].shape)
# # print('x250 dimension: \n ',X[250].ndim)
#
# #  activation functions
# # one : Relu
#
# print('\n ++++++++ Relu activation function ++++++++++++++++++++++++++ \n ')
#
#
# class Activation_Relu:
#     def forward(self, inputs):
#         self.output = np.maximum(0, inputs)
#
#
# activation1 = Activation_Relu()
# activation1.forward(Layer2.output)
# print('Activation Relu = \n', activation1.output)
#
# print('\n ++++++++ Stepup activation function ++++++++++++++++++++++++++ \n ')
#
#
# # 2) stepup action function
# class stepup_activation:
#     def __init__(self):
#         self.output = None
#
#     def forward(self, inputs):
#         self.output = np.heaviside(inputs, 0)
#
#
# activation2 = stepup_activation()
# activation2.forward(Layer2.output)
# print('Activation Out= \n', activation2.output)
# print('Out Activation = \n', activation1.output)
#
# # 3 Softmax activation function
# print('\n ++++++++ Softmax activation function ++++++++++++++++++++++++++ \n ')
#
#
# class softmax_activation:
#     def __init__(self):
#         self.out_values_aftermv = None
#
#     def forward(self, inputs):
#         exp_values = np.exp(inputs)
#         # to remove the overflow
#         exp_values_aftermv = exp_values - np.max(exp_values)
#         self.out_values_aftermv = exp_values_aftermv / np.sum(exp_values_aftermv)
#
#         exp_values_total = np.sum(exp_values)
#         probabilities = exp_values / exp_values_total
#         self.output = probabilities
#
#
# activation3 = softmax_activation()
# activation3.forward(Layer2.output)
# # print('Softmax Activation Function = \n',activation3.output)
# # print('\n ++++++++ Remove the overflow  activation function ++++++++++++++++++++++++++ \n ')
# # print('Aver flow : \n',activation3.exp_values_aftermv)
#
# print('\n ++++++++ Averflow remove Probabilities  activation function ++++++++++++++++++++++++++ \n ')
# print('Averflow remove Probabilities: \n', activation3.out_values_aftermv)
#
#
# class loss:
#     def __init__(self):
#         self.out = None
#
#     def loss_calculator(self, y_pred, y_true):
#         samples = len(y_pred)
#         y_pred = np.clip(y_pred, ep, 1 - ep)
#         if len(y_true.shape) == 1:
#             confidences = y_pred[range(samples), y_true]
#         elif len(y_true.shape) == 2:
#             confidences = np.sum(y_pred * y_true, axis=1, keepdims=1)
#             confidences_log = np.log(confidences)
#             loss = np.mean(confidences_log)
#             self.out = loss
# import matplotlib.pyplot as plt
# import numpy as np
#
# # X=np.random.randn(3,4)
# # X=np.array([2.1,2.2,-1.8])
# # W=[[[5,3.1,2],[2.3,-2.8,3.1]],[[3.1,2.5,3.3],[3.1,2.5,3.3]]]
# # W=[[5,3.1,2],[2.3,-2.8,3.1],[3.1,2.5,3.3],[3.1,2.5,3.3]]
# # W=[[5,3.1],[2.3,-2.8,3.1],[3.1,2.5,3.3],[3.1,2.5,3.3]]
# # W=np.array(W)
# # print(W.shape)
# # print(W.ndim)
# # print(type(X))
# # print(W)
# # print(X.ndim)
# # print(X.shape)
# # print(X)
# # print('Number of dimension for X = ',X.ndim)
#
# # Neuron network with learning
#
#
# input = [2.1, 2.3, 3.4]
# # bias=[3.3,4.4]
# # w1=[[5,3.1],[2.3,-2.8],[3.1,2.5]]
# # w1=np.array(w1)
# # print(w1.ndim)
# # print(w1.shape)
# #
# # y=np.dot(w1.T,input)+bias
# # print(y)
# # print('Traspose = ',w1.T)
# # print('bias',bias.shape)
#
# # Tenser(object) : the something looking
#
# X = np.random.randn(3, 4)
#
#
# # input=[2.1,2.3,3.4]
# class Dense_leyer:
#     def __init__(self, inputs, neurons):
#         self.w1 = 0.2 * np.random.randn(inputs, neurons)
#         self.bias = 0.3 * np.random.randn(1, neurons)
#
#     def forward(self, inputdata):
#         self.output = np.dot(inputdata, self.w1) + self.bias
#         return self.output
#
#
# Layer1 = Dense_leyer(4, 5)
# Layer1.forward(X)
#
# # Layer 2
# Layer2 = Dense_leyer(5, 2)
# Layer2.forward(Layer1.output)
#
# print('\n ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ \n')
# print('First  Dense Layer output = \n', Layer1.output)
# print('\n ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ \n')
# print(' 2nd is the neuron network output = \n', Layer2.output)
# print('\n ++++++++++ Neuron Network of linear end here +++++++++++++++++ \n')
#
# # display image.
# #
# # X=np.load('X.npy')
# # print('x250:\n ',X[250])
# # plt.imshow(X[250])
# # plt.show()
# # print('x250 shape:\n ',X[250].shape)
# # print('x250 dimension: \n ',X[250].ndim)
#
# #  activation functions
# # one : Relu
#
# print('\n ++++++++ Relu activation function ++++++++++++++++++++++++++ \n ')
#
#
# class Activation_Relu:
#     def forward(self, inputs):
#         self.output = np.maximum(0, inputs)
#
#
# activation1 = Activation_Relu()
# activation1.forward(Layer2.output)
# print('Activation Relu = \n', activation1.output)
#
# print('\n ++++++++ Stepup activation function ++++++++++++++++++++++++++ \n ')
#
#
# # 2) stepup action function
# class stepup_activation:
#     def __init__(self):
#         self.output = None
#
#     def forward(self, inputs):
#         self.output = np.heaviside(inputs, 0)
#
#
# activation2 = stepup_activation()
# activation2.forward(Layer2.output)
# print('Activation Out= \n', activation2.output)
# print('Out Activation = \n', activation1.output)
#
# # 3 Softmax activation function
# print('\n ++++++++ Softmax activation function ++++++++++++++++++++++++++ \n ')
#
#
# class softmax_activation:
#     def __init__(self):
#         self.out_values_aftermv = None
#
#     def forward(self, inputs):
#         exp_values = np.exp(inputs)
#         # to remove the overflow
#         exp_values_aftermv = exp_values - np.max(exp_values)
#         self.out_values_aftermv = exp_values_aftermv / np.sum(exp_values_aftermv)
#
#         exp_values_total = np.sum(exp_values)
#         probabilities = exp_values / exp_values_total
#         self.output = probabilities
#
#
# activation3 = softmax_activation()
# activation3.forward(Layer2.output)
# # print('Softmax Activation Function = \n',activation3.output)
# # print('\n ++++++++ Remove the overflow  activation function ++++++++++++++++++++++++++ \n ')
# # print('Aver flow : \n',activation3.exp_values_aftermv)
#
# print('\n ++++++++ Averflow remove Probabilities  activation function ++++++++++++++++++++++++++ \n ')
# print('Averflow remove Probabilities: \n', activation3.out_values_aftermv)
#
#
# class loss:
#     def __init__(self):
#         self.out = None
#
#     def loss_calculator(self, y_pred, y_true):
#         samples = len(y_pred)
#         y_pred = np.clip(y_pred, ep, 1 - ep)
#         if len(y_true.shape) == 1:
#             confidences = y_pred[range(samples), y_true]
#         elif len(y_true.shape) == 2:
#             confidences = np.sum(y_pred * y_true, axis=1, keepdims=1)
#             confidences_log = np.log(confidences)
#             loss = np.mean(confidences_log)
#             self.out = loss
import matplotlib.pyplot as plt
import numpy as np

# Neuron network with learning


input = [2.1, 2.3, 3.4]


class Dense_leyer:
    def __init__(self, inputs, neurons):
        self.w1 = 0.2 * np.random.randn(inputs, neurons)
        self.bias = 0.3 * np.random.randn(1, neurons)

    def forward(self, inputdata):
        self.output = np.dot(inputdata, self.w1) + self.bias
        return self.output


X = np.random.randn(3, 4)

Layer1 = Dense_leyer(4, 5)
Layer1.forward(X)

Layer2 = Dense_leyer(5, 2)
Layer2.forward(Layer1.output)

print('\n ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ \n')
print('First  Dense Layer output = \n', Layer1.output)
print('\n ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ \n')
print(' 2nd is the neuron network output = \n', Layer2.output)
print('\n ++++++++++ Neuron Network of linear end here +++++++++++++++++ \n')


class Activation_Relu:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)
        return self.output


activation1 = Activation_Relu()
activation1.forward(Layer2.output)
print('\n ++++++++ Relu activation function ++++++++++++++++++++++++++ \n ')
print('Activation Relu = \n', activation1.output)

print('\n ++++++++ Stepup activation function ++++++++++++++++++++++++++ \n ')


class stepup_activation:
    def __init__(self):
        self.output = None

    def forward(self, inputs):
        self.output = np.heaviside(inputs, 0)
        return self.output


activation2 = stepup_activation()
activation2.forward(Layer2.output)
print('Activation Out= \n', activation2.output)
print('Out Activation = \n', activation1.output)

print('\n ++++++++ Softmax activation function ++++++++++++++++++++++++++ \n ')


class softmax_activation:
    def __init__(self):
        self.out_values_aftermv = None

    def forward(self, inputs):
        exp_values = np.exp(inputs)
        exp_values_aftermv = exp_values - np.max(exp_values)
        self.out_values_aftermv = exp_values_aftermv / np.sum(exp_values_aftermv)
        exp_values_total = np.sum(exp_values)
        probabilities = exp_values / exp_values_total
        self.output = probabilities


activation3 = softmax_activation()
activation3.forward(Layer2.output)
print('Averflow remove Probabilities: \n', activation3.out_values_aftermv)


class loss:
    def __init__(self):
        self.out = None

    def loss_calculator(self, y_pred, y_true):
        ep = 1e-15
        samples = len(y_pred)
        y_pred = np.clip(y_pred, ep, 1 - ep)
        if len(y_true.shape) == 1:
            confidences = y_pred[range(samples), y_true]
        elif len(y_true.shape) == 2:
            confidences = -np.log(np.sum(y_pred * y_true, axis=1, keepdims=True))
            loss = np.mean(confidences)
            self.out = loss
        return self.out


# Instantiate loss class
loss_calc = loss()

# Modify y_true to be in one-hot encoded format
y_true = np.array([[1, 0], [0, 1], [1, 0]])  # One-h
# ot encoded labels

# Calculate loss
loss_value = loss_calc.loss_calculator(activation3.output, y_true)
print('\n ++++++++ Loss Calculation ++++++++++++++++++++++++++ \n ')
print('Loss Value:', loss_value)
