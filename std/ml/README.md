# Ocean ML v0.1

First PyTorch-style training stack for Ocean.

Core training loop:

```ocean
var model: Linear = Linear(2, 1)
var criterion: MSELoss = MSELoss()
var optimizer: SGD = SGD(model.parameters(), 0.08)

optimizer.zero_grad()

var prediction: Tensor[float32] = model.forward(x)
var loss: Tensor[float32] = criterion.forward(prediction, y)

loss.backward()
optimizer.step()
```

Tensor remains the only tensor object. There is no separate Variable/GradTensor.

Included:
- eager dynamic autograd;
- requires_grad / grad / backward / zero_grad;
- differentiable add/sub/mul/div;
- scalar add/sub/mul/div;
- 2D matmul and transpose;
- ReLU;
- MSE loss;
- Parameter;
- Module;
- Linear;
- ReLU module;
- MSELoss module;
- SGD.

v0.1 limitations:
- float32 autograd only;
- scalar backward only;
- CPU SGD only;
- 2D matmul backward;
- indexing, reshape, slice, copy and to are not differentiable yet;
- graph is freed after backward;
- no retain_graph/no_grad/Adam/CrossEntropy/LayerNorm/DataLoader yet;
- autograd metadata is not thread-safe yet.
