import math


class Item:
    def __init__(self, data, _parents=(), _operator='', label=''):
        self.data = data
        self.label = label
        self._parents = set(_parents)
        self._operator = _operator
        self.grad = 0.0
        self._backward = lambda: None

    def __str__(self):
        return f"Item(data = {self.data})"

    def __repr__(self):
        return f"Item(data = {self.data})"

    def _topological_order(self):
        order = []
        visited = set()

        def helper(node):
            if node not in visited:
                visited.add(node)
                for parent in node._parents:
                    helper(parent)
                order.append(node)

        helper(self)
        return order

    def backwards(self):
        self.grad = 1.0
        order = self._topological_order()
        for node in reversed(order):
            node._backward()

    def __add__(self, other):
        if not isinstance(other, Item):
            other = Item(other)
        out = Item(self.data + other.data, (self, other), "+")

        def _backward():
            self.grad += 1 * out.grad
            other.grad += 1 * out.grad

        out._backward = _backward
        return out

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def __radd__(self, other):
        return self + other

    def __mul__(self, other):
        if not isinstance(other, Item):
            other = Item(other)
        out = Item(self.data * other.data, (self, other), "*")

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __rmul__(self, other):
        return self * other

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only support int/float data types as powers"
        out = Item(self.data ** other, (self,), "pow")

        def _backward():
            self.grad = other * (self.data ** (other - 1)) * out.grad

        out._backward = _backward
        return out

    def __truediv__(self, other):
        return self * (other ** -1)

    def tanh(self):
        x = self.data
        t = (math.exp(2 * x) - 1) / (math.exp(2 * x) + 1)
        out = Item(t, (self,), "tanh")

        def _backward():
            self.grad += (1 - t ** 2) * out.grad

        out._backward = _backward
        return out

    def exp(self):
        x = self.data
        out = Item(math.exp(x), (self,), "exp")

        def _backward():
            self.grad += out.data * out.grad

        self._backward = _backward
        return out
