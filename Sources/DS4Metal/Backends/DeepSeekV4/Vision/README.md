# DeepSeek V4 Flash Vision Experimental

`DeepSeekV4VisionEncoder` loads the separate BF16 vision sidecar published in
`antirez/deepseek-v4-gguf`. It requires the **Vision-Exp language checkpoint**;
the ordinary Flash checkpoint does not have the same language weights.
The GGUF metadata and all 316 tensor names, dimensions and types are validated
before any GPU encoding. The supported source checkpoint revision is
`e46e16bf6035c6f317eb2ac7458eb0362926d402`.

The native pipeline consists of ImageIO image decoding with EXIF orientation,
antialiased bicubic resize and letterboxing, 14×14 channel-major patches,
32 BF16 transformer blocks, 2D RoPE, full image attention, and a 3×3 spatial
downsampling aligner producing 4096-dimensional embeddings. Images are limited
to 40 MiB encoded, 40 megapixels decoded and 16,384 pixels per dimension.
Input is normalized to `pixel / 127.5 - 1`.

Matrix products use Metal Performance Shaders with Float32 inputs and
accumulation. BF16 weights are converted on the GPU into a reusable buffer,
and the reference model's BF16 rounding boundaries are explicitly retained.
The largest temporary conversion buffer is 144 MiB. This avoids loading
Float32 copies of all encoder weights. Individual numerical reductions can
differ from upstream's custom matrix kernels; end-to-end model parity requires
the real sidecar and language checkpoint and is not asserted by synthetic tests.

`promptBlock` packs image patches and learned start/pad/newline/end embeddings
into at most 384 synthetic token IDs. It must receive the **absolute prompt
position**, because padding aligns the start marker to position 3 modulo 4.
The returned `embeddings` and `tokens` have identical row counts. The language
decoder consumes complete blocks, with bidirectional image attention and visual
expert routing; sending these IDs through ordinary text embedding lookup is invalid.

The implementation follows [antirez/ds4](https://github.com/antirez/ds4/tree/9ab705347c1775e7599ede7eb81a6255ec7dccb5),
especially `ds4_image.c`, `ds4_metal.m`, `metal/deepseek4_vision.metal` and
`metal/glm53_vision.metal`. The upstream MIT notice is retained in
`DeepSeekV4VisionKernels.swift`. Tests retain the upstream layout golden vectors
and verify preprocessing, BF16 rounding, MPS matrix orientation, image attention
and spatial unpacking without downloading model weights.
