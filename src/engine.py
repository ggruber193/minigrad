import functools
import math
from typing import Self


class Value:
    def __init__(self, value, _children=(), _op='', label=''):
        self._value = value
        self._children = set(_children)
        self._grad = 0
        self._backward = lambda: None
        self._label = label
        self._op = _op

    @property
    def grad(self):
        return self._grad

    def _topsort(self):
        ordering = []
        stack = [(self, False)]
        visited = set()
        while stack:
            node, processed = stack.pop()
            if node not in visited:
                visited.add(node)
                stack.append((node, True))
                for child in node._children:
                    stack.append((child, False))
            elif processed:
                ordering.append(node)
        return list(reversed(ordering))


    def backward(self):
        ordering = self._topsort()
        self._grad = 1.0
        for node in ordering:
            node._backward()

    def __mul__(self, other):
        if isinstance(other, int) or isinstance(other, float):
            other = Value(other)
        elif isinstance(other, Value):
            pass
        else:
            raise NotImplementedError(f"Only implemented for {self.__class__.__name__}")
        label = f"{self._label}*{other._label}".strip('*')
        out = Value(self._value * other._value, (self, other), "*", label=label)

        def _backward():
            self._grad += other._value * out._grad
            other._grad += self._value * out._grad

        out._backward = _backward
        return out

    def __add__(self, other):
        if isinstance(other, int) or isinstance(other, float):
            other = Value(other)
        elif isinstance(other, Value):
            pass
        else:
            raise NotImplementedError(f"Only implemented for {self.__class__.__name__}")

        label = f"{self._label}+{other._label}".strip('+')
        out = Value(self._value + other._value, (self, other), "+", label=f"({label})")

        def _backward():
            self._grad += out._grad  # 1 * out._grad
            other._grad += out._grad

        out._backward = _backward

        return out

    def exp(self):
        _exp =math.exp(self._value)
        out = Value(_exp, (self, ), "exp")

        def _backward():
            self._grad += _exp * out._grad

        out._backward = _backward
        return out

    def __truediv__(self, other):
        return self ** -1

    def __pow__(self, power, modulo=None):
        if not isinstance(power, int) and not isinstance(power, float):
            raise ValueError("Only implemented for float and int.")
        out = Value(self._value ** power, (self, ), f"**{power}")

        def _backward():
            self._grad += power * self._value ** (power - 1) * out._grad

        out._backward = _backward
        return out

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    def __rsub__(self, other):
        return self - other

    def __rtruediv__(self, other):
        return self / other

    def __repr__(self):
        return f"{self._label} {self._value}".strip()

    def __str__(self):
        return str(self._value)

    def tanh(self):
        _tanh = lambda x: 1 - 2 / (math.exp(2*x)+1)
        t = _tanh(self._value)
        assert abs(t - math.tanh(self._value)) <= 1e-5, f"Custom tanh not matching math tanh: {t} {math.tanh(self._value)}"

        label = f"tanh({self._label})" if self._label else ''
        out = Value(t, (self,), "tanh", label=label)
        def _backward():
            self._grad += (1 - t ** 2) * out._grad

        out._backward = _backward

        return out

    def relu(self):
        _relu = max(0, self._value)
        label = f"relu({self._label})" if self._label else ''
        out = Value(_relu, (self,), "relu")

        def _backward():
            self._grad += (0 if _relu <= 0 else 1) * out._grad

        out._backward = _backward
        return out



if __name__ == '__main__':
    a = Value(2.0)
    print({a,})
    print(max(0, a))
    exit()
    from plot import draw_dot
    # inputs x1,x2
    x1 = Value(2.0, label='x1')
    x2 = Value(0.0, label='x2')
    # weights w1,w2
    w1 = Value(-3.0, label='w1')
    w2 = Value(1.0, label='w2')
    # bias of the neuron
    b = Value(6.8813735870195432, label='b')
    # x1*w1 + x2*w2 + b
    x1w1 = x1 * w1;
    x1w1.label = 'x1*w1'
    x2w2 = x2 * w2;
    x2w2.label = 'x2*w2'
    x1w1x2w2 = x1w1 + x2w2;
    x1w1x2w2.label = 'x1*w1 + x2*w2'
    n = x1w1x2w2 + b;
    n.label = 'n'
    o = n.tanh();
    o.label = 'o'

    o.backward()

    dot = draw_dot(o)
    dot.render()
