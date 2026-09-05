# Getting Started

[README](../README.md)

## Get and build DwarfStar

```sh
git clone https://github.com/antirez/ds4.git
cd ds4
```

Choose your build. The platform pages cover prerequisites and memory limits:

| System | Build | Instructions |
| --- | --- | --- |
| Apple Silicon Mac | `make` | [Metal](METAL.md) |
| DGX Spark / GB10 | `make cuda-spark` | [DGX Spark](DGX_SPARK.md) |
| AMD Strix Halo | `make strix-halo` | [Strix Halo](STRIX_HALO.md) |
| Other NVIDIA GPUs, including multiple cards | `make cuda-generic` | [CUDA GPUs](CUDA_MULTI_GPU.md) |

Run commands in these guides from the repository root, not from `docs/`.
The build produces `ds4`, `ds4-agent`, `ds4-server`, `ds4-bench`, and `ds4-eval`.

## Download a model

For a first run on a 96 or 128 GB machine, start with DeepSeek V4 Flash Q2:

```sh
./download_model.sh ds4f-q2
```

The script stores weights in `gguf/` and points `ds4flash.gguf` at the selected
main model. Downloading another main model changes that link. Use `-m FILE`
when selecting a model explicitly.

Run the same download command again to resume an interrupted download.
Some targets require the Hugging Face CLI; the script prints its installation
command if it is missing. Authentication is optional for public weights.
It uses your cached Hugging Face token or `HF_TOKEN` when present.

The file size is not the total memory requirement. Context, scratch buffers,
vision, draft models, and extra server sessions all need memory too.
On a smaller Mac, use [SSD streaming](SSD_STREAMING.md).

## Run it

```sh
./ds4
./ds4 -p "Explain Redis streams in one paragraph."
./ds4-agent
```

The selected build uses its GPU backend by default. Thinking is enabled;
add `--nothink` for direct answers. Start with the default context, then use
`--ctx N` when you need a larger window.

For a local API:

```sh
./ds4-server --ctx 32768
```

The default endpoint is `http://127.0.0.1:8000`. See [serving](SERVER.md) for
multiple sessions and [client setup](CLIENTS.md) for external coding agents.

Next: [other models and vision](MODELS.md),
[speculative decoding](SPECULATIVE_DECODING.md), or
[two-machine inference](DISTRIBUTED.md).
