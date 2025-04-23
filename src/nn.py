
from engine import Value
import random

class Neuron:
    def __init__(self, n_in):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(n_in)]
        self.b = Value(random.uniform(-1, 1))

    def __call__(self, x):
        a = sum((w_i * x_i for w_i, x_i in zip(self.w, x)), self.b)
        z = a.tanh()
        return z

    def parameters(self):
        return self.w + [self.b]

class Layer:
    def __init__(self, n_in, n_out):
        self._n_in = n_in
        self._n_out = n_out
        self.neurons = [Neuron(n_in) for _ in range(n_out)]

    def __call__(self, x):
        outs = [neuron(x) for neuron in self.neurons]
        return outs[0] if len(outs) == 1 else outs

    def __repr__(self):
        return f"{self._n_in} X {self._n_out}"

    def parameters(self):
        return [j for i in self.neurons for j in i.parameters()]

class MLP:
    def __init__(self, n_in: int, n_outs: list[int]):
        in_outs = [n_in] + n_outs
        self.layers = [Layer(in_outs[i], in_outs[i+1]) for i in range(len(in_outs)-1)]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def __repr__(self):
        return '\n'.join([str(i) for i in self.layers])

    def parameters(self):
        return [j for i in self.layers for j in i.parameters()]


if __name__ == '__main__':
    mlp = MLP(3, [4, 4, 1])
    xs = [
        [2.0, 3.0, -1.0],
        [3.0, -1.0, 0.5],
        [0.5, 1.0, 1.0],
        [1.0, 1.0, -1.0],
    ]
    ys = [1.0, -1.0, -1.0, 1.0]  # desired targets
    y_pred = [mlp(i) for i in xs]

    lr = 0.05
    lr_decay = 0.99

    for epoch in range(100):
        y_pred = [mlp(i) for i in xs]
        loss = sum(((y_hat-y)**2 for y, y_hat in zip(ys, y_pred)))

        for p in mlp.parameters():
            p._grad = 0.0
        loss.backward()

        print(loss)

        for p in mlp.parameters():
            p._value -= lr * p.grad

        lr = lr * lr_decay
