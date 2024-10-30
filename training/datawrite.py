# typstifier
# training/datawrite.py

# Store model in binary format

import jax.numpy as jnp
import flax.nnx as nnx
import numpy as np
import struct
import model as m
import pickle

# Filepaths

in_filepath = "checkpoints/modelcheck-100000.pkl"
out_filepath = "checkpoints/model-weights.bin"

# Taken from https://github.com/karpathy/llama2.c/blob/master/export.py

def serialize_fp32(out_file, arr):
    """Write flattened array to file in wb mode"""
    d = np.array(jnp.reshape(arr, -1), dtype = "float32")
    b = struct.pack(f"{len(d)}f", *d)
    out_file.write(b)

def write_model(out_filepath, model):
    out_file = open(out_filepath, "wb")

    serialize_fp32(out_file, model.conv1.kernel)
    serialize_fp32(out_file, model.conv1.bias)
    serialize_fp32(out_file, model.conv2.kernel)
    serialize_fp32(out_file, model.conv2.bias)
    serialize_fp32(out_file, model.lin1.kernel)
    serialize_fp32(out_file, model.lin1.bias)
    serialize_fp32(out_file, model.lin2.kernel)
    serialize_fp32(out_file, model.lin2.bias)

    out_file.close()

model = m.ConvNet(rngs = nnx.Rngs(42))
with open(in_filepath, "rb") as f:
    out = pickle.load(f)
nnx.update(model, out["model"])
write_model(out_filepath, model)
