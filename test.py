import framework.modules.trainable_modules.dense_layer
import framework.modules.criterion_modules.mse_layer
import framework.modules.activation_modules.relu_layer
import framework.modules.activation_modules.tanh_layer
import framework.modules.sequential

import torch
import numpy as np

#Construc a simple model (D->D->A->M)
layers = []
dense_layer = framework.modules.trainable_modules.dense_layer.DenseLayer(4, 3, True)
dense_layer_2 = framework.modules.trainable_modules.dense_layer.DenseLayer(3, 1, True)

dense_layerS = framework.modules.trainable_modules.dense_layer.DenseLayer(2, 1, True)
dense_layer_2S = framework.modules.trainable_modules.dense_layer.DenseLayer(1, 1, True)

relu_layer = framework.modules.activation_modules.relu_layer.ReLuLayer()#tanh_layer.TanhLayer()
dense_layer_out = framework.modules.trainable_modules.dense_layer.DenseLayer(3, 1, True)
relu_layer_out = framework.modules.activation_modules.relu_layer.ReLuLayer()#tanh_layer.TanhLayer()#relu_layer.ReLuLayer()
mse_layer = framework.modules.criterion_modules.mse_layer.MSELayer()
x=np.array([[1,0,0,0],[1,0,0,0],[0,0,0,0]])
layers.append(dense_layer)
layers.append(dense_layer_2)
layers.append(mse_layer)
seq = framework.modules.sequential.Sequential(layers)
#Output
y=np.array([[1],[1],[0]])


X = torch.from_numpy(x)
X = X.type(torch.FloatTensor)
Y = torch.from_numpy(y)
Y = Y.type(torch.FloatTensor)
epoch = 50
for i in range(epoch):
    memory = []
    #loss = seq.forward(X, Y)
    #print("seq loss", loss)
    #seq.backward(X, Y)
#exit()


#Construct a simple model. (D->A->D->A->M)
layers = []
dense_layer = framework.modules.trainable_modules.dense_layer.DenseLayer(4, 3, True)
relu_layer = framework.modules.activation_modules.tanh_layer.TanhLayer() #relu_layer.ReLuLayer()#
dense_layer_out = framework.modules.trainable_modules.dense_layer.DenseLayer(3, 1, True)
relu_layer_out = framework.modules.activation_modules.tanh_layer.TanhLayer()#relu_layer.ReLuLayer()

dense_layerS = framework.modules.trainable_modules.dense_layer.DenseLayer(4, 3, True)
relu_layerS = framework.modules.activation_modules.tanh_layer.TanhLayer() #relu_layer.ReLuLayer()#
dense_layer_outS = framework.modules.trainable_modules.dense_layer.DenseLayer(3, 1, True)
relu_layer_outS = framework.modules.activation_modules.tanh_layer.TanhLayer()#relu_layer.ReLuLayer()
mse_layer = framework.modules.criterion_modules.mse_layer.MSELayer()
layers.append(dense_layerS)
layers.append(relu_layerS)
layers.append(dense_layer_outS)
layers.append(relu_layer_outS)
layers.append(mse_layer)
model = framework.modules.sequential.Sequential(layers)
x=np.array([[1,0,0,0],[0,0,0,1],[0,0,0,0]])

#Output
y=np.array([[1],[1],[0]])


X = torch.from_numpy(x)
X = X.type(torch.FloatTensor)
Y = torch.from_numpy(y)
Y = Y.type(torch.FloatTensor)
epoch = 300

#D->A->D->A->M
for i in range(epoch):
    m0 = X
    m1 = dense_layer.forward(m0) #D
    a_m1 = relu_layer.forward(m1) #A

    m2 = dense_layer_2.forward(a_m1) #D
    a_m2 = relu_layer_out.forward(m2) #A

    loss = mse_layer.forward(a_m2, Y) #M
    print("n loss:", loss)
    loss = model.forward(m0, Y)
    print("s loss:", loss)
    print("---")

    memory = [m0, m1, a_m1, m2, a_m2]
    e0 = mse_layer.backward(a_m2, Y) #M

    slope_out = relu_layer_out.backward(m2) #A
    c_e0 = e0 * slope_out

    e1 = dense_layer_2.backward(c_e0) #D
    w = torch.t(a_m1).mm(c_e0)
    b = torch.sum(c_e0)
    dense_layer_2.apply_gradient(w, b, 0.2)

    slope_hid = relu_layer.backward(m1) #A
    c_e1 = e1 * slope_hid

    e2 = dense_layer.backward(c_e1) #D -useless-
    w = torch.t(m0).mm(c_e1)
    b = torch.sum(c_e1) #sum bias??? on error? Dimension msimactch
    dense_layer.apply_gradient(w, b, 0.2)


    model.backward(a_m2, Y)
exit()

dtype = torch.FloatTensor
# dtype = torch.cuda.FloatTensor # Uncomment this to run on GPU

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 64, 1000, 100, 10

# Create random input and output data
x = torch.randn(N, D_in).type(dtype)
y = torch.randn(N, D_out).type(dtype)

# Randomly initialize weights
w1 = torch.randn(D_in, H).type(dtype)
w2 = torch.randn(H, D_out).type(dtype)

learning_rate = 1e-6
for t in range(500):
    # Forward pass: compute predicted y
    h = x.mm(w1)
    h_relu = h.clamp(min=0)
    y_pred = h_relu.mm(w2)

    # Compute and print loss
    loss = (y_pred - y).pow(2).sum()
    print(t, loss)

    # Backprop to compute gradients of w1 and w2 with respect to loss
    grad_y_pred = 2.0 * (y_pred - y)
    grad_w2 = h_relu.t().mm(grad_y_pred)
    grad_h_relu = grad_y_pred.mm(w2.t())
    grad_h = grad_h_relu.clone()
    grad_h[h < 0] = 0
    grad_w1 = x.t().mm(grad_h)

    # Update weights using gradient descent
    w1 -= learning_rate * grad_w1
    w2 -= learning_rate * grad_w2