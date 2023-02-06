# NanoAutoGrad
A miniature implementation of PyTorch's autograd

## Installation
```
 pip install NanoAutoGrad
```

## Usage:
***Calculating gradients of a function w.r.t input***
```python
from NanoAutoGrad.engine import Item

a = Item(3)
b = Item(5)
c = Item(10)

d = a * b
e = d + c
# backpropagating to calculate gradients w.r.t leaf nodes
e.backwards()
print(f"gradient w.r.t a: {a.grad}")
print(f"gradient w.r.t b: {b.grad}")
print(f"gradient w.r.t c: {c.grad}")

# ------- Outputs ------- 
# gradient w.r.t a: 5
# gradient w.r.t b: 3
# gradient w.r.t c: 1
```

***Training small neural network***
```python
import matplotlib.pyplot as plt
from NanoAutoGrad.network import MLP
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

data = load_iris()
features = data.data
targets = data.target
x_train, x_test, y_train, y_test = train_test_split(features, targets, train_size=0.8, shuffle=True)

# initializing model
model = MLP(4, [16, 16, 1])
lr = 0.01

# training loop
epochs = 100
losses = []
for e in range(epochs):
    epoch_loss = 0.0
    for x, y in zip(x_train, y_train):
        x = x.tolist()
        preds = model(x)
        loss = (y - preds) ** 2 / len(y_train)
        epoch_loss += loss.data
        for p in model.parameters():
            p.grad = 0.0
        loss.backwards()
        for p in model.parameters():
            p.data += -1 * lr * p.grad
    losses.append(epoch_loss)
    print(f"epoch: {e} | loss: {epoch_loss}")

plt.plot(range(epochs), losses)
plt.show()

# ------- Outputs ------- 
# epoch: 0 | loss: 9.632519370429291
# epoch: 1 | loss: 4.01727203958246
# epoch: 2 | loss: 3.0380594863430743
# epoch: 3 | loss: 2.3873438447464355
# epoch: 4 | loss: 1.922751222912399
# epoch: 5 | loss: 1.5807488839288562
# epoch: 6 | loss: 1.3234329739835267
# epoch: 7 | loss: 1.1264671408938376
# epoch: 8 | loss: 0.9735081358648532
# epoch: 9 | loss: 0.8532107038929909
# .....
```
> <img src="https://d2beiqkhq929f0.cloudfront.net/public_assets/assets/000/024/811/original/myplot.png?1675671242">
