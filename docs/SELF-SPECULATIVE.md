**English** | [Italiano](SELF-SPECULATIVE.it.md)

# Experimental self-speculative CLI decode

> **Status as of July 13, 2026:** phases A-C are implemented in the CLI demo and
> enabled only via `DS4_SPEC_K`; they are not integrated into the GUI path or
> the `InferenceService`. The 2026-07-08 runs on an M1 Pro 16 GB with the Flash
> IQ2_XXS GGUF verified greedy parity on the prompts tried, but throughput came
> out below normal decode. The feature is therefore opt-in and parked, not a
> recommended profile. All numbers below are snapshots of that configuration,
> not a promise for other models or Macs.

Goal of the experiment: break decode's "one full pass per token" structure —
route → selection readback → gather → FFN, ×43 layers — by paying it
**once every N tokens**. In the 3.33 tok/s snapshot of 2026-07-08 the local
levers looked exhausted: gather at 87-94% of the SSD ceiling, sync at 3 ms,
routed FFN hidden by ASYNC_FFN.
The GGUF tried had no MTP weights; the variant below does not require them.

## Scheme

```
state S at position P (last accepted token t0)
1. SNAPSHOT   s = specSnapshot()                  (~1-2 MB, recurrent state only)
2. DRAFT      K CHEAP forwards (activeExperts=2) from P: candidates c1..cK
              (they write raw KV at positions P..P+K-1 and advance the
               compressors with "draft" values — these will be REWRITTEN by
               verification)
3. RESTORE    specRestore(s)
4. VERIFY     one full-config BATCH step over tokens [t0? no: c1..cK] with
              PER-POSITION logits: reuses the prefill's batched phase A (route
              of K tokens in ONE command buffer) + dedup of the expert union
              (batchedExpertLayer). Rewrites raw KV and full-config compressor
              state for P..P+K-1.
5. ACCEPT     greedy: j = length of the prefix ci == argmax(logits[i-1]);
              the logit of the LAST verified position gives the next token
              FOR FREE (bonus token): advancement = j+1 tokens per round.
6. ROLLBACK   if j < K: specRestore(s) and batch RE-VERIFY of just the j
              accepted tokens (mini-step, rebuilds clean state at P+j).
              If j == K: the verification state IS already the right one.
```

Initial cost estimate per round: 1 batch verify (≈ 1 full pass, amortized
over j+1 tokens) + K drafts at ~1/3 of the expert I/O + (only on rejection) a
mini-verify of the accepted
tokens. The expectation was **1.5-2.5×** with average acceptance 2-3; later
measurements disproved the assumption of a sufficiently cheap draft.

## Constraints verified against the code

- **NSA recurrence** (the constraint that makes "rewinding" impossible and
  the reason for the engine's `kvDirty`): per layer, `CompressorState.stateKv`
  + `stateScore` ([coff·ratio × width] f32, ~32 KB at ratio-4) + `count`,
  doubled for layers with an indexer (`indexStates`). Rollback does NOT
  require the cache rows: emission indexes by `count`, so rows written beyond
  the restored count get rewritten (by the full-config verification) before
  any read. → LIGHTWEIGHT snapshot (`SpecRecurrentState`),
  ~1-2 MB, not the ~22 KB/token of the checkpoints' `KVSnapshot`.
- **Raw KV**: positional (ring `pos % rawRows`): the rows of rejected
  positions are overwritten by subsequent tokens; no active rollback.
- **activeExperts**: already runtime in the dispatch (`min(d.activeExperts,
  d.k)` in DecodeLayer, route weights renormalized); it only needed to be
  made mutable on the decoder (`setActiveExperts`, Phase A).
- **Top-K indexer**: activates at a deterministic threshold of compressed
  rows — with contexts where the DIAG proves it cannot activate
  (`DS4_LAZY_IDX`), draft and verify stay on the dense, coherent path. With
  the indexer active, batch verification falls back to phase A's per-token
  path (already handled).

## Correctness

- Greedy: the accepted tokens are EXACTLY the ones the full model would have
  generated (position-by-position argmax comparison) — bit-for-bit parity
  with normal decode, by construction.
- Sampling: requires rejection sampling (accept ci with prob min(1,
  p_full(ci)/p_draft(ci)), otherwise resample from the residual
  distribution). LATER PHASE: we start greedy-only.
- Repetition penalty and tool parsing operate on ACCEPTED tokens — the
  speculative window stays internal to the engine.

## Incremental plan

1. **Phase A (primitives) — DONE**: runtime
   `setActiveExperts(_:)`/`activeExpertsNow`; lightweight snapshot/restore of
   the recurrent state (`SpecRecurrentState`, `SpecDecode.swift`). No
   behavior change while unused.
2. **Phase B (verify) — DONE**: `specVerifyStep(tokens:startPos:) ->
   [[Float]]` (`Sources/DS4Metal/Backends/DeepSeekV4/Decode/Prefill/StreamingDecoder+Prefill.swift`,
   next to `prefillRange`, whose
   batched stage and expert union it reuses): full-config batch step with
   per-position logits (output head on every final hidden, ~8 ms/token).
3. **Phase C (loop, demo) — DONE and measured on limited coverage**:
   `DS4_SPEC_K=N`
   (+`DS4_SPEC_DRAFT_EXPERTS`, default 2) in the demo: greedy
   draft/verify/accept loop with bonus token and rebuild on rejection.
   Permanent validation must compare the same text as normal decode across
   multiple prompts and sweep K=2..6, reading tokens/round and acceptance.
4. **Phase D (GUI/engine) — not started, parked**: integration into
   InferenceService (greedy generate → speculative when temperature==0 or
   behind a toggle), acceptance telemetry
   in the profile (tokens/round), then rejection sampling.
   Integration note: ALWAYS restore activeExperts on error
   paths (defer), and the draft marginally pollutes the usage imatrix
   (routes recorded with draft selection) — acceptable, possibly gate it.

## First field measurement

Snapshot 2026-07-08: M1 Pro 16 GB, Flash IQ2_XXS, K=4, 2-expert draft.

- **Parity: PERFECT** — text identical character for character to normal
  decode over 48 tokens. NSA recurrence snapshot/rollback, verification and
  acceptance all work (draft acceptance 49%, 2.53 tokens/round).
- **Economics: currently LOSES** (0.79 vs 3.13 tok/s). Two measured causes:
  1. verification went through `batchedExpertLayer`, which at small windows
     re-reads the UNION from disk ignoring the slot-cache (692 vs 477
     MB/token) → FIXED: `specVerifyStep` now uses the layer-major per-token
     path (hits from the pool, dense weights amortized once per layer);
  2. the DRAFT is not ~1/3 of the cost as estimated: with 2 experts only the
     routed gather shrinks, but every forward pays the full streamed dense
     pass (~0.9 GB/pass with QKV_Q4, without SHARED_Q4) and the per-layer
     overhead → a draft forward ≈ 0.8× a full forward, and with K=4 that is
     3 extra passes per round.
- **For it to win, the draft must cost ≤0.2× a forward**: (a)
  `DS4_SHARED_Q4=1` (takes the shared FFNs out of the stream: benefits
  normal decode too, to be re-measured now that q_a/kv/trio are resident);
  (b) a draft that SKIPS the shared FFN (it is an approximation anyway: it
  only affects acceptance); (c) small K (2) to reduce draft passes per
  round; (d) looking ahead, layer-skip in the draft or a future MTP
  integration with a dedicated draft head. Today the DIAG only detects their
  presence: the current path neither loads nor uses those weights.

## Second measurement (same snapshot, verify via slot-cache + SHARED_Q4)

- Parity again perfect; K=2: **78% acceptance**, 1.78 tokens/round;
  K=4: 50%, 2.40 tokens/round. Average forward down to 224 ms (from the
  verification fix: 308 vs 692 MB/token from disk).
- **Still losing** (K=2: 2.38 vs 3.36 tok/s for the same-knob baseline).
  The cause is STRUCTURAL, not implementation: after the session's
  optimizations the per-token cost is dominated by SERIAL route/attn (~60%),
  which batch verification does not amortize — every position pays its own
  attention; only the dense weights (now small) and the gather (now cached)
  are amortized. Speculation pays off when the token is dominated by
  amortizable weight I/O: that was true at 0.47 tok/s, no longer at 3.4.
- CONCLUSION: parked as an experimental opt-in (DS4_SPEC_K).
  It becomes worthwhile again if/when: (a) verification gets true
  MULTI-TOKEN route/attn (causal flash-attn over the window in one dispatch,
  matvec→matmul for K rows); or (b) per-token route/attn drops a lot
  (micro-chain fusion). To be re-evaluated after those workstreams.

## Phase N — prompt-lookup (n-gram) draft: measured, parked

> **Measurement 2026-07-14** (M1 Pro 16 GB, Flash IQ2_XXS, fast preset, K=4,
> incremental-list prompt, 100 tokens): PERFECT parity with normal decode;
> throughput 1.86 vs 3.08 tok/s. Two causes:
> 1. **Structural, decisive**: even FULLY accepted rounds (+4 tokens) ran at
>    2.64-2.95 tok/s, BELOW the baseline — batch verification costs ~1
>    forward PER POSITION (serial route/attn, as already measured for
>    Phase C). With verify(K) ≈ K forwards, NO draft —
>    not even a free one — can win: the theoretical ceiling is ~1.0× plus
>    just the bonus token, and rejections (re-verify of the prefix) push it
>    below. Profile: 187 positions computed for 100 tokens emitted.
> 2. The incremental-numbers prompt is ADVERSARIAL for prompt-lookup: the
>    pattern ", numero " repeats but the number that follows changes each
>    time (39% acceptance, rounds at +2). On literal repetition acceptance
>    rises, but because of cause 1 it still is not enough.
> CONCLUSION: experimental opt-in like Phase C. Any speculation on this
> engine becomes worthwhile ONLY with true multi-token verification (causal
> flash-attn over the window in one dispatch + matvec→matmul for K rows) —
> that is the workstream that unlocks the whole family.

With Phase M blocked on the artifact (below), the only ~zero-cost draft
available TODAY is the transcript itself: `PromptLookup.draft`
(`Sources/DS4Core/Generation/`) finds the most recent occurrence of the
current suffix (4→2 n-gram) in prompt+generated text and proposes the tokens
that followed it. Demo: `DS4_SPEC_K=N DS4_SPEC_DRAFT=ngram`. Key property
compared to the reduced-experts draft: when the lookup finds nothing, the
round degrades to a NORMAL forward (no snapshot, no verification) — the path
only pays where there is actually something to copy. Same full-config batch
verification, same greedy acceptance, same parity by construction; per-round
telemetry: acceptance + miss count. Honest expectation: the gain exists only
on repetitive text (code, tool output, prompt quotations — the agentic use
case), because verification stays per-token on the route/attn; on novel
prose the behavior converges to normal decode.

## Phase V — multi-token verification (in progress)

The conclusion of measurements C and N is the same: with `verify(K) ≈ K
forwards` no draft can win. This phase attacks verification.

1. **V1 (DONE, to be measured)** — `DS4_SPEC_VERIFY_BATCH` (default on):
   specVerifyStep enqueues the whole window's route/attention in ONE
   command buffer per layer (encodeRouteInto + snapshot blit, the batched
   prefill's phase A: one sync per layer instead of 3·K cbs) and serves the
   routed FFNs from the per-token SLOT-CACHE (hits from the pool — the
   reason verification avoided batchedExpertLayer and its from-disk union).
   FFNs are serialized at token boundaries (join before the acquire:
   back-to-back acquires on the same layer with FFNs in flight could evict
   slots still being read by the GPU). Same dispatches in the same order per
   token: identical numerics; ineligible layers fall back to per-token.
   HONEST expected gain: synchronization/orchestration only (~3-8%): the
   route's GPU work stays per-token — V1 is above all the SCAFFOLDING for V2.
2. **V2 (to do)** — matvec→matmul over the window: the route's dense
   projections (q/kv/out/shared, today re-read K times) become [K × inDim]
   matmuls that read the weights ONCE (existing kernels:
   dense kernel_mul_mm_q8_0/f16; for the resident Q4_K reuse
   kernel_mul_mm_id_q4_K_f32 with single expert id=0, the same trick as the
   dense matvec). Requires splitting encodeRouteInto into pre-attn
   (batchable) / attention (sequential over the KV) / post-attn (batchable
   after a barrier). This is where the bulk of the gain is.
3. **V3 (to do)** — causal flash-attention over the window in one
   dispatch (existing batched dk512 kernel + blk prepass + causal
   mask): closes the remainder.

## Phase M — draft with the MTP head (BLOCKED on the artifact)

> **Status 2026-07-14:** the MTP component registered as an internal
> accessory (`DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf`) does NOT appear to
> be published on the HF repo `antirez/deepseek-v4-gguf`, and the main Flash
> IQ2_XXS GGUF contains no nextn/mtp weights (verified with `DS4_DIAG=1`:
> "MTP: nessun peso nel GGUF"). Phase M1 (below) remains ready: if/when the
> artifact appears, the inventory report is the first step. Producing the
> sidecar ourselves would require the original DeepSeek-V4 release and a
> dedicated conversion.

The second measurement closed out the reduced-experts draft: it costs ~0.8×
a full forward (pays the streamed dense weights and per-layer overhead in
full) and the structural ceiling remains the serial route/attn that
verification does not amortize. DeepSeek's MTP head is the right draft: ONE
transformer block + head (~1/43 of the layers per candidate) trained exactly
for the next token — expected acceptance well above the reduced draft's
49-78%. The sidecar has an internal accessory descriptor (id `mtp`, outside
the GUI's main model catalog;
`DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf`, ~4 GB, Flash only); DeepSeek
interface: `h' = block(eh_proj([enorm(emb(t)); hnorm(hidden)]))` →
`shared_head(norm(h'))`, iterated K-1 times reusing `h'`.

1. **M1 (DONE)**: `MTPSidecar`
   (`Sources/DS4Metal/Backends/DeepSeekV4/MTP/`) — opening, role
   classification (eh_proj, embed_tokens, enorm, hnorm,
   shared_head.*, block) and a validation report against the main model's
   vocab/nEmbd. Demo: `DS4_MTP_GGUF=<path>` (or `=1` to look for
   `*MTP*.gguf` next to the model). No effect on decode: it produces the
   ground truth (the sidecar's REAL names/shapes/quants) to wire up M2
   without guessing the conversion.
2. **M2**: resident load (~4 GB, RAM-gated) + draft forward.
   Requires: the main model's final pre-head hidden exposed by the decoder
   (outputHead's `oembd`, before `out.norm`); the MTP block's own KV (its
   attention sees the window's positions); the exact block wiring decided by
   the M1 report (if the block is a complete DSV4 layer, reuse `decodeLayer`
   with dedicated `LayerWeights`, otherwise dedicated kernels).
3. **M3**: hook into the demo loop (the MTP draft chain replaces the
   reduced-experts chain inside `DS4_SPEC_K`), same verification/acceptance;
   sweep K=2..4, character-for-character greedy parity, acceptance
   telemetry.

Honest expectation (from the second measurement): the MTP draft alone ≈
1.2-1.3× because verification stays per-token on the route/attn; the 1.5-2×
package also requires true multi-token verification (causal flash-attn over
the window + matvec→matmul for K rows).

## Risks and mitigations

- Dirty state after rejection → rebuild mini-verify (step 6); K=1 parity as
  a permanent test.
- "Cheap" draft KV read by verification? NO: verification recomputes and
  rewrites ALL positions of the window starting from the snapshot; the draft
  reads its own KV (self-attention internal to the draft window) —
  consistent with the self-speculative scheme (the draft is a "different"
  model).
- Low acceptance on hard text → adaptive K (reduce K when average
  acceptance drops; already planned in the Phase C loop).
- Bandwidth: the draft adds ~K/3 of expert I/O per round; if acceptance is
  ≥1.5 the bytes-per-accepted-token balance stays favorable. The round DIAG
  (tokens/round, bytes/token) measures it.
