# typstifier
# training/train.py

import numpy as np
import jax.numpy as jnp
from flax import nnx
import matplotlib.pyplot as plt
import pickle
import optax
import model as m

np.random.seed(41)

# Parameters

total = 100_000
batch_size = 32
lr = 1e-3
weight_decay = 1e-2
norm_clipping = 1

print_frequency = 10_000
alpha = 0.99
plot_step = 100

input_filename = "numpydata.pkl"
output_filename = f"checkpoints/modelcheck-{total}.pkl"
plot_filename = f"checkpoints/training-{total}.png"

# Model definitions

model = m.ConvNet(rngs = nnx.Rngs(42))
gradient_transform = optax.chain(
    optax.clip_by_global_norm(norm_clipping),
    optax.adamw(lr, weight_decay = weight_decay),
)

optimizer = nnx.Optimizer(model, gradient_transform)

@nnx.jit
def train_step(model, optimizer, x, y):
    loss, grads = nnx.value_and_grad(m.loss_fn)(model, x, y)
    optimizer.update(grads)
    return loss, optax.tree_utils.tree_l2_norm(grads)

@nnx.jit
def eval_step(model, x, y):
    return m.loss_fn(model, x, y)

# Load data

with open(input_filename, "rb") as f:
    (xs_all, ys_all) = pickle.load(f)

img_size = xs_all.shape[1]
total_img_num = xs_all.shape[0]
total_sym_num = ys_all.shape[1]

def unison_shuffled(a, b):
    rng_state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(rng_state)
    np.random.shuffle(b)

def prepare_batch(xs, ys, u, v):
    len_tot = xs.shape[0]
    u_trunc, v_trunc = u % len_tot, v % len_tot

    if u_trunc == 0 and u > 0:
        unison_shuffled(xs, ys)
    
    if v_trunc == u_trunc + (v - u):
        xs_o = xs[u_trunc:v_trunc, ...]
        ys_o = ys[u_trunc:v_trunc, ...]
    else:
        xs_first = xs[u_trunc:, ...]
        ys_first = ys[u_trunc:, ...]
        unison_shuffled(xs, ys)
        xs_second = xs[:v_trunc, ...]
        ys_second = ys[:v_trunc, ...]
        xs_o = np.concatenate([xs_first, xs_second])
        ys_o = np.concatenate([ys_first, ys_second])
       
    return jnp.array(xs_o), jnp.array(ys_o)

unison_shuffled(xs_all, ys_all)
u, v = int(0.8 * total_img_num), int(0.9 * total_img_num)
xs_train, ys_train = xs_all[:u, ...], ys_all[:u, ...]
xs_val, ys_val = xs_all[u:v, ...], ys_all[u:v, ...]
xs_test, ys_test = xs_all[v:, ...], ys_all[v:, ...]

print(f"Image size: {img_size}x{img_size}")
print(f"Dataset sizes: train {u}, val {v - u}, test {total_img_num - v}")
print(f"Number of symbols: {total_sym_num}")
print(f"Cross entropy loss for random selection: {np.log(total_sym_num):.3f}")

# Train

indices = [batch_size * (i + 1) for i in range(total)]
train_losses = []
train_losses_eval = []
val_losses = []
grad_norms = []

model.train() # Use dropout

for i in range(total):
    u, v = i * batch_size, (i + 1) * batch_size
    xs_t, ys_t = prepare_batch(xs_train, ys_train, u, v)
    xs_v, ys_v = prepare_batch(xs_val, ys_val, u, v)

    new_train_loss, new_grad_norm = train_step(model, optimizer, xs_t, ys_t)
    
    model.eval() # Disable dropout in this block
    new_train_loss_eval = eval_step(model, xs_t, ys_t)
    new_val_loss = eval_step(model, xs_v, ys_v)
    model.train()

    if i == 0:
        train_losses.append(new_train_loss)
        train_losses_eval.append(new_train_loss_eval)
        val_losses.append(new_val_loss)
        grad_norms.append(new_grad_norm)
    else:
        train_losses.append(alpha * train_losses[-1] + (1 - alpha) * new_train_loss)
        train_losses_eval.append(alpha * train_losses_eval[-1] + (1 - alpha) * new_train_loss_eval)
        val_losses.append(alpha * val_losses[-1] + (1 - alpha) * new_val_loss)
        grad_norms.append(alpha * grad_norms[-1] + (1 - alpha) * new_grad_norm)
    
    if (i + 1) % print_frequency == 0:
        print(
            f"Run {(i + 1) * batch_size}:",
            f"train loss {train_losses[-1]:.3f},",
            f"train eval loss {train_losses_eval[-1]:.3f},",
            f"val loss {val_losses[-1]:.3f},",
            f"grad norm {grad_norms[-1]:.3f}"
        )
   
# Report test error

loop_num = xs_test.shape[0] // batch_size
test_loss = 0

model.eval() # Disable dropout

for i in range(loop_num):
    u, v = i * batch_size, (i + 1) * batch_size
    xs, ys = prepare_batch(xs_test, ys_test, u, v)
    test_loss += eval_step(model, xs, ys)

test_loss /= loop_num

print(f"Final: test loss {test_loss:.5f}")

# Plot losses and gradient norms

fig, (ax_l, ax_r) = plt.subplots(1, 2)
ax_l.plot(indices[::plot_step], train_losses[::plot_step], label = "train")
ax_l.plot(indices[::plot_step] * np.arange(0, total, plot_step), train_losses_eval[::plot_step], label = "train eval")
ax_l.plot(indices[::plot_step] * np.arange(0, total, plot_step), val_losses[::plot_step], label = "val")
ax_r.plot(indices[::plot_step] * np.arange(0, total, plot_step), grad_norms[::plot_step], label = "grad norm")
ax_l.legend()
ax_r.legend()
plt.savefig(plot_filename)

# Example output to compare with inference code

x = 1e-4 * np.arange(0, 64 * 64)
x = np.reshape(x, (1, 64, 64, 1))
x = nnx.softmax(model(x))
ordered = list(range(total_sym_num))
ordered.sort(key = lambda i: x[0, i], reverse = True)        
print(f"Example output: {ordered[:10]}")

# Save checkpoint

out = {
    "model": nnx.state(model),
    "test_loss": test_loss,
    "indices": indices,
    "train_losses": train_losses,
    "train_losses_eval": train_losses_eval,
    "val_losses": val_losses,
    "grad_norms": grad_norms
}

with open(output_filename, "wb") as f:
    pickle.dump(out, f)
