// typstifier
// inference/src/main.rs

use itertools::iproduct;
use wasm_bindgen::prelude::*;

// Weights and parameters

const MODEL_WEIGHTS: &[u8] = include_bytes!("../../training/checkpoints/model-weights.bin");

const DRAW_SIZE: usize = 64;

const CONV1_WINDOW: usize = 7;
const CONV1_IN: usize = 1;
const CONV1_OUT: usize = 32;

const POOL1: usize = 4;

const CONV2_WINDOW: usize = 3;
const CONV2_IN: usize = 32;
const CONV2_OUT: usize = 64;

const POOL2: usize = 2;

const LIN1_IN: usize = 8 * 8 * 64;
const LIN1_OUT: usize = 64;

const LIN2_IN: usize = 64;
const LIN2_OUT: usize = 839;

// Binding for logging

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = console, js_name = log)]
    fn _log(s: &str);
}

// Tensor datastructures and layers

struct Tensor {
    data: Vec<f32>,
    dims: Vec<usize>,
    prods: Vec<usize>,
}

fn iterated_prods(dims: &Vec<usize>) -> (Vec<usize>, usize) {
    let mut dims = dims.clone();
    dims.reverse();
    let mut prods: Vec<usize> = dims
        .iter()
        .scan(1, |acc, x| {
            *acc *= x;
            Some(*acc)
        })
        .collect();
    let total_prod = prods.pop().unwrap();
    prods.reverse();
    prods.push(1);
    (prods, total_prod)
}

impl Tensor {
    fn from_bytes(all_data: &[u8], offset: &mut usize, dims: Vec<usize>) -> Self {
        let (prods, total_prod) = iterated_prods(&dims);
        let data = all_data[(4 * *offset)..(4 * *offset + 4 * total_prod)]
            .chunks_exact(4)
            .map(|chunk| {
                let array: [u8; 4] = chunk.try_into().unwrap();
                f32::from_le_bytes(array)
            })
            .collect();
        let out = Tensor { data, dims, prods };
        *offset += total_prod;

        out
    }

    fn from_floats(all_data: &[f32], dims: Vec<usize>) -> Self {
        let (prods, _) = iterated_prods(&dims);
        Tensor {
            data: Vec::from(all_data),
            dims,
            prods,
        }
    }

    fn zeros(dims: Vec<usize>) -> Self {
        let (prods, total_prod) = iterated_prods(&dims);
        Tensor {
            data: vec![0.0; total_prod],
            dims,
            prods,
        }
    }

    fn flatten(&mut self) {
        let (_, total_prod) = iterated_prods(&self.dims);
        self.dims = vec![total_prod];
        self.prods = vec![1, total_prod];
    }

    #[inline(always)]
    fn get_index(&self, dims: &Vec<usize>) -> usize {
        self.prods
            .iter()
            .zip(dims.iter())
            .map(|(&a, &b)| a * b)
            .sum()
    }

    #[inline(always)]
    fn access(&self, dims: &Vec<usize>) -> f32 {
        self.data[self.get_index(dims)]
    }

    #[inline(always)]
    fn update(&mut self, dims: &Vec<usize>, val: f32) {
        let idx = self.get_index(dims);
        self.data[idx] = val
    }
}

fn conv_layer(kernel: &Tensor, bias: &Tensor, x_in: &Tensor) -> Tensor {
    assert!(kernel.dims.len() == 4 && bias.dims.len() == 1 && x_in.dims.len() == 3);
    let (kernel_w, kernel_h, f_in, f_out) = (
        kernel.dims[0],
        kernel.dims[1],
        kernel.dims[2],
        kernel.dims[3],
    );
    let (width, height) = (x_in.dims[0], x_in.dims[1]);
    assert!(x_in.dims[2] == f_in && bias.dims[0] == f_out);

    // https://openxla.org/xla/operation_semantics#convwithgeneralpadding_convolution
    // x_out[i, j, k] = x_in[i + u - kernel_w / 2, j + v - kernel_h / 2, k] * kernel[u, v, w, k] + bias[k]
    let mut x_out = Tensor::zeros(vec![width, height, f_out]);
    for (i, j, k) in iproduct!(0..width, 0..height, 0..f_out) {
        let mut curr_sum = bias.access(&vec![k]);
        for (u, v, w) in iproduct!(0..kernel_w, 0..kernel_h, 0..f_in) {
            let val = if i + u >= kernel_w / 2
                && j + v >= kernel_h / 2
                && i + u < width + kernel_w / 2
                && j + v < height + kernel_h / 2
            {
                x_in.access(&vec![i + u - kernel_w / 2, j + v - kernel_h / 2, w])
            } else {
                0.0
            };
            curr_sum += kernel.access(&vec![u, v, w, k]) * val;
        }
        x_out.update(&vec![i, j, k], curr_sum.max(0.0));
    }
    x_out
}

fn max_pool(x_in: &Tensor, window: usize) -> Tensor {
    assert!(x_in.dims.len() == 3 && x_in.dims[0] % window == 0 && x_in.dims[1] % window == 0);
    let mut x_out = Tensor::zeros(vec![
        x_in.dims[0] / window,
        x_in.dims[1] / window,
        x_in.dims[2],
    ]);

    for (i, j, k) in iproduct!(0..x_out.dims[0], 0..x_out.dims[1], 0..x_out.dims[2]) {
        let mut curr_max = f32::NEG_INFINITY;
        for (u, v) in iproduct!(0..window, 0..window) {
            curr_max = curr_max.max(x_in.access(&vec![i * window + u, j * window + v, k]));
        }
        x_out.update(&vec![i, j, k], curr_max);
    }
    x_out
}

fn linear_layer(kernel: &Tensor, bias: &Tensor, x_in: &Tensor, relu: bool) -> Tensor {
    assert!(kernel.dims.len() == 2 && bias.dims.len() == 1 && x_in.dims.len() == 1);
    let (f_in, f_out) = (kernel.dims[0], kernel.dims[1]);
    assert!(x_in.dims[0] == f_in && bias.dims[0] == f_out);

    let mut x_out = Tensor::zeros(vec![f_out]);

    // x_out[i] = x_in[j] * kernel[j, i] + bias[i] (summation convention)
    for i in 0..f_out {
        let mut curr_sum = bias.access(&vec![i]);
        for j in 0..f_in {
            curr_sum += x_in.access(&vec![j]) * kernel.access(&vec![j, i]);
        }
        if relu {
            curr_sum = curr_sum.max(0.0);
        }
        x_out.update(&vec![i], curr_sum);
    }
    x_out
}

fn softmax(x: &mut Tensor) {
    let max_val = x
        .data
        .iter()
        .fold(f32::NEG_INFINITY, |curr_max, &val| curr_max.max(val));
    x.data
        .iter_mut()
        .for_each(|val_ptr| *val_ptr = (*val_ptr - max_val).exp());
    let sum: f32 = x.data.iter().sum();
    x.data
        .iter_mut()
        .for_each(|val_ptr| *val_ptr = *val_ptr / sum);
}

// Run inference code

#[wasm_bindgen]
pub fn infer_symbol(drawing: &[f32]) -> Vec<usize> {
    let mut offset = 0;

    let conv1_kernel = Tensor::from_bytes(
        &MODEL_WEIGHTS,
        &mut offset,
        vec![CONV1_WINDOW, CONV1_WINDOW, CONV1_IN, CONV1_OUT],
    );
    let conv1_bias = Tensor::from_bytes(&MODEL_WEIGHTS, &mut offset, vec![CONV1_OUT]);

    let conv2_kernel = Tensor::from_bytes(
        &MODEL_WEIGHTS,
        &mut offset,
        vec![CONV2_WINDOW, CONV2_WINDOW, CONV2_IN, CONV2_OUT],
    );
    let conv2_bias = Tensor::from_bytes(&MODEL_WEIGHTS, &mut offset, vec![CONV2_OUT]);

    let lin1_kernel = Tensor::from_bytes(&MODEL_WEIGHTS, &mut offset, vec![LIN1_IN, LIN1_OUT]);
    let lin1_bias = Tensor::from_bytes(&MODEL_WEIGHTS, &mut offset, vec![LIN1_OUT]);

    let lin2_kernel = Tensor::from_bytes(&MODEL_WEIGHTS, &mut offset, vec![LIN2_IN, LIN2_OUT]);
    let lin2_bias = Tensor::from_bytes(&MODEL_WEIGHTS, &mut offset, vec![LIN2_OUT]);

    assert_eq!(4 * offset, MODEL_WEIGHTS.len());

    let x = Tensor::from_floats(&drawing, vec![DRAW_SIZE, DRAW_SIZE, CONV1_IN]);
    let x = conv_layer(&conv1_kernel, &conv1_bias, &x);
    let x = max_pool(&x, POOL1);
    let x = conv_layer(&conv2_kernel, &conv2_bias, &x);
    let mut x = max_pool(&x, POOL2);
    x.flatten();
    let x = linear_layer(&lin1_kernel, &lin1_bias, &x, true);
    let mut x = linear_layer(&lin2_kernel, &lin2_bias, &x, false);

    // Whether or not we compute the softmax here does not matter but we leave it in so we can show
    // confidence percentages later.

    softmax(&mut x);

    let mut indices: Vec<usize> = (0..x.data.len()).collect();
    indices.sort_by(|&i, &j| x.data[j].partial_cmp(&x.data[i]).unwrap());
    indices
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_inference() {
        let drawing: Vec<f32> = (0..(64 * 64)).map(|i| 1e-4 * (i as f32)).collect();
        let indices = infer_symbol(&drawing);
        // Run test with --nocapture to print output
        println!("{:?}", &indices[..10]);
    }
}
