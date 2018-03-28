import framework.modules.trainable_modules.dense_layer
import framework.modules.criterion_modules.mse_layer
import framework.modules.activation_modules.relu_layer
import framework.modules.sequential

import torch
import numpy as np

#Construc a simple model (D->D->A->M)
layers = []
dense_layer = framework.modules.trainable_modules.dense_layer.DenseLayer(4, 1, True)
dense_layer_2 = framework.modules.trainable_modules.dense_layer.DenseLayer(1, 1, True)

dense_layerS = framework.modules.trainable_modules.dense_layer.DenseLayer(4, 1, True)
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
epoch = 20
for i in range(epoch):
    memory = []
    loss = seq.forward(X, Y)
    print("seq loss", loss)

    memory.append(X)
    m1 = dense_layerS.forward(X)
    memory.append(m1)
    m2 = dense_layer_2S.forward(m1)
    memory.append(m2)
    loss = mse_layer.forward(m2, Y)
    print(loss)
    print('----')

    e0 = mse_layer.backward(m2, Y)

    e1 = dense_layer_2S.backward(e0)
    w = torch.t(memory[1]).mm(e0)
    b = torch.sum(e0)
    dense_layer_2S.apply_gradient(w, b, 0.1)

    e2 = dense_layerS.backward(e1)
    w = torch.t(memory[0]).mm(e1)
    b = torch.sum(e1)
    dense_layerS.apply_gradient(w, b, 0.1)

    seq.backward(X, Y)
exit()



#Construct a simple model. (D->A->D->A->M)
layers = []
dense_layer = framework.modules.trainable_modules.dense_layer.DenseLayer(4, 3, True)
relu_layer = framework.modules.activation_modules.relu_layer.ReLuLayer()#tanh_layer.TanhLayer()
dense_layer_out = framework.modules.trainable_modules.dense_layer.DenseLayer(3, 1, True)
relu_layer_out = framework.modules.activation_modules.relu_layer.ReLuLayer()#tanh_layer.TanhLayer()#relu_layer.ReLuLayer()
mse_layer = framework.modules.criterion_modules.mse_layer.MSELayer()
layers.append(dense_layer)
layers.append(relu_layer)
layers.append(dense_layer_out)
layers.append(relu_layer_out)
layers.append(mse_layer)
model = framework.modules.sequential.Sequential(layers)
x=np.array([[1,0,1,0],[1,0,1,1],[0,1,0,1]])

#Output
y=np.array([[1],[1],[0]])


X = torch.from_numpy(x)
X = X.type(torch.FloatTensor)
Y = torch.from_numpy(y)
Y = Y.type(torch.FloatTensor)
epoch = 20

#D->A->D->A->M
for i in range(epoch):
    hid_l = dense_layer.forward(X) #D
    hid_l_a = relu_layer.forward(hid_l) #A

    out_l = dense_layer_out.forward(hid_l_a) #D
    out_l_a = relu_layer_out.forward(out_l) #A

    loss = mse_layer.forward(out_l_a, Y) #M

    #loss = model.forward(X, Y)
    #out_l_a = model.memory[3]
    #hid_l_a = model.memory[1]

    E = mse_layer.backward(out_l_a, Y) #M

    slope_out = relu_layer_out.backward(out_l_a) #A
    d_out = E * slope_out
    E_hid = dense_layer_out.backward(d_out) #D

    slope_hid = relu_layer.backward(hid_l_a) #A
    d_hid = E_hid * slope_hid
    E_s = dense_layer.backward(d_hid) #D -useless-

    wo_grad = hid_l_a.mm(d_out)
    bo_grad = torch.sum(d_out)
    wh_grad = torch.t(X).mm(d_hid)
    bh_grad = torch.sum(d_hid)


    dense_layer_out.apply_gradient(wo_grad, bo_grad, 0.1)
    dense_layer.apply_gradient(wh_grad, bh_grad, 0.1)

    print(loss)
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