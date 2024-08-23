# typstifier

The goal of this project is to write a character classifier which outputs the command needed in
[Typst](https://typst.app/) to generate the drawn symbol. The project is currently under active
development and not functional yet.

# Setup

You will need to have the current version of [Rust](https://www.rust-lang.org/) and
[Node](https://nodejs.org/) as well as [NPM](https://www.npmjs.com/) installed. To generate the WASM
binary as well as the glue code, install `wasm-bingen-cli` using `cargo install`, make sure its
version matches the one in `inference/Cargo.toml` and run

```bash
cd inference
cargo build --release --target=wasm32-unknown-unknown
wasm-bindgen --out-dir ../frontend/src/lib/pkg target/wasm32-unknown-unknown/release/inference.wasm
```

To install the NPM packages and start the development server, execute the commands below.

```bash
cd frontend
npm install
npm run dev
```

Later versions will contain a step to generate the training data and execute the training loop (as
well as implement a working inference algorithm and frontend).

## License

This project is licensed under the MIT License.
