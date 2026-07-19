**English** | [Italiano](README-analisi.it.md)

# GGUF compressibility analysis

This guide describes the two exploratory tools used to measure weight
redundancy before investing in new kernels, factorized representations, or
fine-tuning. Both read the GGUF without modifying it and do not produce a
model that DwarfStar can run directly.

Back to the [scripts index](README.md).

## Requirements

```sh
python3 -m pip install -U gguf numpy
```

The `gguf` package provides dequantization for the formats supported by
`gguf.quants`. The commands below assume they are run from the repository
root.

## `gguf_spectrum.py`: spectrum and redundancy

For the selected tensors the script computes the singular value spectrum and
reports the effective rank at 90%, 95%, and 99% of the energy. For routed
experts it additionally estimates, via projection and PCA, how many shared
bases are needed to explain the variability across experts.

```sh
python3 scripts/gguf_spectrum.py /percorso/modello.gguf
python3 scripts/gguf_spectrum.py /percorso/modello.gguf --layers 0,2,20 --experts
python3 scripts/gguf_spectrum.py /percorso/modello.gguf --full
python3 scripts/gguf_spectrum.py /percorso/modello.gguf --layers 2 --json risultati.json
```

`--full` also includes the embeddings and the output head and can take much
more time and memory. The expert estimates depend on `--proj`: they are a
decision-making tool, not a proof of model equivalence.

## `gguf_to_graph.py`: graph factorization

Each selected matrix `W[out x in]` is approximated with a truncated SVD:

```text
input -- B[r x in] --> bottleneck r -- A[out x r] --> output
```

The parameters live on the edges `A` and `B`; the nodes describe only the
activation spaces. The script can produce JSON, Graphviz DOT, and, on request,
a NumPy archive with the factors.

```sh
python3 scripts/gguf_to_graph.py /percorso/modello.gguf \
  --layers 2 --energy 0.95 --dot grafo.dot --json grafo.json

dot -Tsvg grafo.dot -o grafo.svg

python3 scripts/gguf_to_graph.py /percorso/modello.gguf \
  --layers 0,2 --experts --rank 64 --npz fattori.npz
```

The transformation is **lossy**. The summary reports the fraction of
parameters retained and the relative reconstruction error; recovering quality
normally requires training or fine-tuning followed by end-to-end validation.

## Choosing the GGUF

The engine handles expert tensors in `Q4_K`, `Q2_K`, and `IQ2_XXS`, dense
tensors in `Q8_0`, and `F16`/`F32` values where needed. For a detailed expert
analysis it is best to start from a less aggressively quantized GGUF:
conclusions drawn from a 2-bit model do not automatically transfer to other
variants.

## Correct interpretation

- A numerically low rank does not guarantee that linguistic quality remains
  unchanged.
- Results depend on the layer, the tensor, the quantization, and the
  analysis's sampling parameters.
- Before implementing a new format, compare memory, bandwidth, logit error,
  and quality on a reproducible set of prompts.
- These scripts are offline tools: they are not part of the load or inference
  path of the GUI or the demo.
