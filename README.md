# typstifier

The goal of this project is to write a character classifier which outputs the command needed in
[Typst](https://typst.app/) to generate the drawn symbol. The project is currently under active
development and not functional yet.

# Setup

In order to build this project, you will need to have [Rust](https://www.rust-lang.org/),
[Typst](https://typst.app/), [ImageMagick](https://imagemagick.org/) and [Node](https://nodejs.org/)
as well as [NPM](https://www.npmjs.com/) installed.

The generation of the training images is done with following commands. This will download the
relevant Rust file from the Typst project, extract the symbol names from it, generate a JSON file
and the training images within the folder `data` and save the icons for the frontend in
`frontend/static/symbols`.

```bash
cd databuild
wget -O symrs.txt https://raw.githubusercontent.com/typst/typst/main/crates/typst/src/symbols/sym.rs
cargo run symrs.txt ../data ../frontend/static/symbols
```

To generate the WASM binary as well as the glue code, install `wasm-bingen-cli` using `cargo
install`, make sure its version matches the one in `inference/Cargo.toml` and run

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

Later versions will contain a step to execute the training loop (and implement a working inference
algorithm and frontend).

## License

This project is licensed under the MIT License.
