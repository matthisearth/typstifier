# typstifier
# training/model.py

import flax.nnx as nnx
from functools import partial
import optax

class ConvNet(nnx.Module):
    def __init__(self, rngs: nnx.Rngs):
        self.conv1 = nnx.Conv(1, 32, kernel_size = (7, 7), padding = "same", rngs = rngs)
        self.conv2 = nnx.Conv(32, 64, kernel_size = (3, 3), padding = "same", rngs = rngs)

        self.max_pool1 = partial(nnx.max_pool, window_shape = (4, 4), strides = (4, 4))
        self.max_pool2 = partial(nnx.max_pool, window_shape = (2, 2), strides = (2, 2))

        self.lin1 = nnx.Linear(8 * 8 * 64, 64, rngs = rngs)
        self.lin2 = nnx.Linear(64, 839, rngs = rngs)

        self.dropout1 = nnx.Dropout(0.2, rngs = rngs)
        self.dropout2 = nnx.Dropout(0.2, rngs = rngs)
        self.dropout3 = nnx.Dropout(0.2, rngs = rngs)
        self.dropout4 = nnx.Dropout(0.5, rngs = rngs)

    def __call__(self, x):
        # (64, 64, 1) (first batch dimension omitted)
        x = self.max_pool1(self.dropout1(nnx.relu(self.conv1(x))))
        # (16, 16, 32)
        x = self.max_pool2(self.dropout2(nnx.relu(self.conv2(x))))
        # (8, 8, 64)
        x = x.reshape(x.shape[0], -1)
        # (8 * 8 * 64)
        x = self.dropout4(nnx.relu(self.lin1(x)))
        # (64)
        x = self.lin2(x)
        # (825)
        return x

def loss_fn(model, x, y):
    y_pred = model(x)
    return optax.losses.softmax_cross_entropy(y_pred, y).mean()
