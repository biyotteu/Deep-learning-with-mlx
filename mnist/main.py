from dataset.mnist import load_mnist
import numpy as np
import models.mlp as model

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

def loss_fn(model, X, y):
    return mx.mean(nn.losses.cross_entropy(model(X),y))

def eval_fn(model, X, y):
    return mx.mean(mx.argmax(model(X), axis=1) == y)

def iter(batch_size, X, y):
    perm = mx.array(np.random.permutation(y.size))
    for s in range(0, y.size, batch_size):
        ids = perm[s:s+batch_size]
        yield X[ids], y[ids]

(x_train, t_train),(x_test,t_test) = load_mnist(flatten=True,normalize=True)

# model params
num_layers = 2
hidden_dim = 32
num_classes = 10
batch_size = 128
num_epochs = 100
learning_rate = 1e-1

print(x_train.shape)
model = model.MLP(num_layers=num_layers, input_dim=x_train.shape[-1], hidden_dim=hidden_dim, output_dim=num_classes)
mx.eval(model.parameters())

loss_and_grad = nn.value_and_grad(model, loss_fn)

optimizer = optim.SGD(learning_rate=learning_rate)

max_acc = 0.0

for e in range(num_epochs):
    for X, y in iter(batch_size=batch_size,X=x_train,y=t_train):
        loss, grads = loss_and_grad(model,X,y)

        optimizer.update(model, grads)

        mx.eval(model.parameters(), optimizer.state)
    
    acc = eval_fn(model, x_test, t_test)
    print(f"Epoch {e}: Test accuracy {acc.item():.3f}")
    if acc > max_acc:
        model.save_weights("best.npz")
