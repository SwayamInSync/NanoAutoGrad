# NanoAutoGrad
A miniature implementation of PyTorch's autograd

## Installation
```
 pip install NanoAutoGrad
```

## Usage:
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
```
