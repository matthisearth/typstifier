# typstifier
# Makefile

.PHONY: databuild training inference

databuild:
	cd databuild && \
	wget -O symrs.txt https://raw.githubusercontent.com/typst/typst/0.11/crates/typst/src/symbols/sym.rs && \
	cargo run symrs.txt ../data ../frontend/static/symbols.png ../frontend/src/lib/names.json && \
	cd ..

training:
	cd training && \
	python3 dataload.py && \
	python3 train.py && \
	python3 datawrite.py && \
	cd ..

inference:
	cd inference && \
	cargo build --release --target=wasm32-unknown-unknown && \
	wasm-bindgen --out-dir ../frontend/src/lib/pkg target/wasm32-unknown-unknown/release/inference.wasm && \
	cd ..
