# GGUF Compression Analysis

Two Python tools help reason from measured numbers before investing in new
kernels, compression strategies, or fine-tuning work. They are especially useful
for exploring quantization and graph-style representations where weights live on
edges.

```sh
pip install -U gguf numpy        # llama.cpp's gguf package provides correct dequantization
```

## 1. `gguf_spectrum.py`: How Compressible Is It?

For the main tensors, this script computes the **singular-value spectrum** and
reports the effective rank at 90/95/99% retained energy, plus the potential
parameter savings from low-rank factorization.

For **experts**, it estimates **redundancy across the 256 experts** with PCA over
expert matrices. That tells you how many shared bases might be enough before a
quality recovery pass.

```sh
python3 scripts/gguf_spectrum.py model.gguf                      # layer 2, dense tensors
python3 scripts/gguf_spectrum.py model.gguf --layers 0,2,20 --experts
python3 scripts/gguf_spectrum.py model.gguf --full              # also output/embeddings; slower
```

The script only reads and measures. It does not modify the model. Use the output
to decide *where* factorization is worth trying.

## 2. `gguf_to_graph.py`: Moving Weights From Nodes To Edges

This script turns matrices into a **factorized graph**. Each `W[out x in]` becomes
a path:

```text
in -- B[r x in] --> bottleneck r -- A[out x r] --> out
```

The factorization is truncated SVD, so `W ~= A * B`. Parameters live on the A/B
edges, while nodes are parameter-free activation spaces. The tool can emit the
graph as JSON or Graphviz DOT, optionally write factors as `.npz`, and report the
compression summary plus **reconstruction error**.

```sh
python3 scripts/gguf_to_graph.py model.gguf --layers 2 --energy 0.95 --dot graph.dot
dot -Tsvg graph.dot -o graph.svg
python3 scripts/gguf_to_graph.py model.gguf --layers 0,2 --experts --rank 64 --npz factors.npz
```

The factorization is **lossy**. The script prints the relative error for each
matrix; recovering model quality would require fine-tuning. This tool builds and
measures the graph representation, but it does not produce a ready-to-run model.

## Format Notes

The Swift engine executes expert tensors in **Q4_K / Q2_K / IQ2_XXS**, dense
tensors in **Q8_0**, plus F16/F32 where needed. The analysis scripts can
dequantize any format supported by `gguf.quants`, but for high-detail expert
analysis the **q4** GGUF is usually the best input because experts remain Q4_K.
The 2-bit model is already close to the edge of the supported formats.
