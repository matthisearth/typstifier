# typstifier
# Makefile

.PHONY: databuild inference

databuild:
	cd databuild && \
	wget -O symrs.txt https://raw.githubusercontent.com/typst/typst/main/crates/typst/src/symbols/sym.rs && \
	cargo run symrs.txt ../data ../frontend/static/symbols ../frontend/src/lib/names.json && \
	cd ..

inference:
	cd inference && \
	cargo build --release --target=wasm32-unknown-unknown && \
	wasm-bindgen --out-dir ../frontend/src/lib/pkg target/wasm32-unknown-unknown/release/inference.wasm && \
	cd ..
