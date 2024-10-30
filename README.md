# typstifier

The goal of this project is to write a character classifier which outputs the command needed in
[Typst](https://typst.app/) to generate the drawn symbol. The project is currently under active
development and not functional yet.

# Setup

In order to build this project, you will need to have [Rust](https://www.rust-lang.org/),
[Typst](https://typst.app/), [ImageMagick](https://imagemagick.org/) and [Node](https://nodejs.org/)
as well as [NPM](https://www.npmjs.com/) installed.

The generation of the training images is done by running `make databuild` which will download the
relevant Rust file from the Typst project, extract the symbol names from it, generate a JSON file
and the training images within the folder `data` and save the icons for the frontend in
`frontend/static/symbols`.

To train the neural network in the `training` directory, you will need to have
[JAX](https://github.com/google/jax) with proper GPU support installed. Moreover, this project uses
[NumPy](https://numpy.org/), [Matplotlib](https://matplotlib.org/), [OpenCV](https://opencv.org/),
[Optax](https://github.com/google-deepmind/optax) and [Flax](https://github.com/google/flax) which
you can install via PIP.

```bash
pip install numpy matplotlib opencv-python optax flax
```

The file `training/dataload.py` writes the training data to the file `training/numpydata.pkl`, the
file `training/train.py` contains the training loop and `training/datawrite.py` writes the
model checkpoint into a binary file. All these scripts can be run in order using `make training`.

To generate the WASM binary as well as the glue code, you will need to install `wasm-bingen-cli`
using `cargo install`, make sure its version matches the one in `inference/Cargo.toml` and run
`make inference`. Finally, in the `frontend` directory, you can install the Node packages using 
`npm install`, start the development server using `npm run dev` and build the project as a static
site using `npm run build`.

## License

This project is licensed under the MIT License.
