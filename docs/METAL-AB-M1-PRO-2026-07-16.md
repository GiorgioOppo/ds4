# Metal kernel A/B on M1 Pro — July 16, 2026

This is an experimental snapshot, not a promise valid for every Apple GPU. The
runs were performed on a MacBook Pro with M1 Pro and 16 GB, using the local
Flash model of about 91 GB, `DS4_RAW_RING=1`, context 4096, a 133-token
prompt, greedy decode of 16 tokens and two warm-up tokens. Expert bundle and
MetalIO were disabled to prevent a sidecar not matching the model from
skewing the comparison.

Each knob was tried in two separate processes and then repeated with the
order inverted. For each pair the runner compared tokens, argmax, top-3 and
2,197,760 Float32 logits with zero tolerance.

| Candidate | Correctness in both orders | Prefill, candidate vs base | Decode profile, candidate vs base | Balanced decode mean | Decision |
|---|---|---:|---:|---:|---|
| `DS4_FLASH_KV_STAGE=1` | `PASS_EXACT` | +3.5% / +1.4% | -1.8% / +2.3% | about +0.2% | Opt-in: small prefill gain, decode neutral |
| `DS4_VECTOR_COPY=1` | `PASS_EXACT` | 0.0% / +11.7% | -3.5% / 0.0% | about -1.8% | Opt-in: noisy prefill, slightly worse decode |
| `DS4_ROPE_PAIR=1` with affine | `PASS_EXACT` | -14.4% / -0.3% | -2.6% / +1.4% | about -0.7% | Opt-in: no end-to-end advantage on M1 Pro |

The balanced mean is the arithmetic mean of the baseline and candidate tok/s
in the two orders, followed by their ratio. It does not remove all the noise,
but it prevents automatically attributing the advantage of the first or
second process to the kernel. The prefill anomalies in `VECTOR_COPY` and
`ROPE_PAIR` show why a single pair is not enough to change the defaults.

## Numeric gate and GPU tests

Beyond the full model, the GPU tests cover:

- raw KV ring with wrap, compressed cache and NSA mask;
- partial and full FlashAttention block;
- F32/F16 conversions with scalar tails and transport of all F16 bits;
- baseline, pair and affine RoPE in decode, prefill, inverse mode and YaRN.

The reproducible local gate is:

```sh
DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer \
  swift test --disable-sandbox \
  --filter 'GraphKVStageABTests|MetalCopyTests|MetalRoPETests'
```

To repeat a full-model comparison use `scripts/metal_ab.sh`, first with the
default order and then inverted:

```sh
DS4_AB_ATOL=0 DS4_AB_RTOL=0 \
  scripts/metal_ab.sh model.gguf prompt.txt DS4_FLASH_KV_STAGE 0 1 16 out/base-first

DS4_AB_ATOL=0 DS4_AB_RTOL=0 DS4_AB_ORDER=candidate-first \
  scripts/metal_ab.sh model.gguf prompt.txt DS4_FLASH_KV_STAGE 0 1 16 out/candidate-first
```

The three paths remain available for measurements on M3/M4 and longer
contexts, but they are not applied automatically by the demo or the GUI until
a balanced A/B on the target machine shows a repeatable advantage.
