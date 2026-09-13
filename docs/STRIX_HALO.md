# AMD Strix Halo

[README](../README.md) | [Getting started](../README.md#start-here)

The reference system is a 128 GB Strix Halo with Radeon 8060S (`gfx1151`),
such as the Framework Desktop. The ROCm build uses the standard binary names
and selects the ROCm backend by default.

## Prerequisites

For a container setup, see the maintained
[ROCm toolbox](https://github.com/kyuz0/strix-halo-ds4-toolbox/blob/main/toolboxes/Dockerfile.rocm-10.0).
It can also be managed with
[AI Toolbox Cockpit](https://github.com/kyuz0/ai-toolbox-cockpit).

For a native Ubuntu build you need HIP, hipBLAS, hipBLASLt, rocBLAS, rocWMMA,
and hipCUB development files. The Ubuntu 26.04 setup used these packages:

```sh
sudo apt-get update
sudo apt-get install -y hipcc rocminfo rocm-smi \
  libamdhip64-dev libhipblas-dev libhipblaslt-dev librocblas-dev \
  librocwmma-dev libhipcub-dev
sudo usermod -aG render,video "$USER"
```

Log out and back in after changing groups. `rocminfo` must report `gfx1151`
and be able to open `/dev/kfd` before DwarfStar can run.

Some packaged rocWMMA headers omit `rocwmma/internal/`. If compilation fails
there, install the complete headers matching your ROCm installation, or use
the container. Do not mix header versions as a general workaround.

## GPU-visible memory

Check the GPU-visible memory pool reported by `rocminfo`. Some 128 GB systems expose only about 62 GiB to the GPU. The tested 128 GB Fedora Linux Strix Halo system, running a recent kernel and ROCm 10.0, used these boot parameters:

```text
amd_iommu=off amdgpu.gttsize=126976 ttm.pages_limit=32505856
```

The GTT/TTM settings expose about 124 GiB to the GPU. An SSD expert-cache request such as `92GB` is fitted to that GPU-visible limit as well as available system RAM; a stock ~62 GiB pool can therefore yield a much smaller cache. `amd_iommu=off` was part of the tested setup, but is not required for GTT sizing and disables DMA isolation. Keep RAM available for the OS. See the [host configuration guide](https://strix-halo-toolboxes.com/#config) for Fedora, Ubuntu/Debian, and systemd-boot instructions.

## Build and run Flash

```sh
make strix-halo
./download_model.sh ds4f-q2
./ds4 --rocm
```

`make rocm` is an alias. Use the current 0731 Q2 download for a first run;
larger mixed and Q4 models have substantially higher memory requirements.
Flash's ROCm resident and pipeline paths should not be confused with the GLM
SSD-streaming path.

## DeepSeek V4.1 Flash

The ROCm 10.0 build supports calibrated V4.1 Flash Q2 text and vision on `gfx1151`. A single 128 GB system was tested with SSD streaming, including a 94 GiB expert/staging cache at 16K text context and in image/state checks. Engram tables remain disk-backed even when expert weights are resident. Cache admission depends on available memory, context size and concurrent sessions. Automatic sizing remains conservative; 94 GiB is a tested manual setting, not a universal maximum. The GPU GTT limit shares physical RAM with the OS and is not itself the usable cache budget.

```sh
make strix-halo ROCM_ARCH=gfx1151
./download_model.sh ds41f-q2
./ds4 --rocm -m gguf/DeepSeek-V4.1-Flash-Q2.gguf --ssd-streaming --ssd-streaming-cache-experts 92GB --ctx 262144
```

The larger-context configuration above allocated 262,144 tokens and completed a real 65,536-token text prompt plus 128 greedy outputs on the 128 GB SSD system. The same test passes in resident mode, with all 129,280 frontier logits and the printed continuation identical. This validates 256K allocation and 64K use; populated 256K inference and retrieval quality were not tested. The tested 92 GiB cache leaves at least 14.10 GiB RAM available at this context allocation. Other workloads and image inputs may need a smaller cache.

Use the matching V4.1 vision sidecar with `--vision FILE`. See [models and vision](MODELS.md#deepseek-v41-flash) for downloads and [qualification results](../QA_BEFORE_RELEASES.md#deepseek-v41-flash-rocmgfx1151) for output quality, numerical drift and memory limitations. Resident text and vision inference were also tested on upcoming 192 GB hardware; performance results will be released soon.

## GLM 5.3 Flash

The reference Q2 setup uses SSD streaming to leave room for its graph and KV
state. Begin with automatic cache sizing and a small context:

```sh
./download_model.sh glm53-q2
./ds4 --rocm -m gguf/GLM-5.3-Flash-Q2.gguf \
  --ssd-streaming --ctx 4096
```

GLM 5.2 also supports ROCm streaming. Full-model GLM 5.2 inference requires it;
distributed layer slices can be resident. See [SSD streaming](SSD_STREAMING.md)
before adjusting the cache budget.

Both GLM 5.3 Flash and DeepSeek Flash Vision Experimental support images on
ROCm. Add the matching encoder with `--vision FILE`, as described in
[models and vision](MODELS.md#vision).

For a model-free routed-kernel check, use `make test-mxfp4-rocm`.
Full-model validation is described in [testing](TESTING.md).
