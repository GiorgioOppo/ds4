# Gap Analysis: DeepSeek-V4-Pro-MacOS vs llama.cpp

Confronto strutturato tra il nostro progetto e
[ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp), il
runtime di riferimento per inferenza locale di LLM.

**Premessa importante**: i due progetti hanno missioni **diverse**,
non sovrapponibili. Questa analisi non è un giudizio di valore — è
un inventario di cosa fa l'uno e non l'altro, utile per (a) capire
dove posizionarsi, (b) decidere se valga la pena chiudere alcuni gap,
(c) riconoscere le aree dove siamo già più avanti.

| Asse | DeepSeek-V4-Pro-MacOS | llama.cpp |
|---|---|---|
| Missione | Runtime verticale per **una** architettura (DeepSeek-V4) sul GUI macOS nativo | Runtime orizzontale general-purpose, **N** architetture, **M** piattaforme |
| Linguaggio | Swift + Metal Shading Language | C/C++ + GGML + N backend HW |
| Piattaforme | macOS 14+ / Apple Silicon | Linux / macOS / Windows / iOS / Android / Docker / Web |
| Modello target | DeepSeek-V4-Flash / V4-Pro | 60+ famiglie di modelli |
| Frontend | SwiftUI desktop app + CLI | CLI + HTTP server + 15+ bindings |
| LOC stimate | ~30 k Swift + ~5 k MSL | ~500 k C/C++ |

---

## 1. Architetture di modello supportate

| | Noi | llama.cpp |
|---|---|---|
| Modelli supportati | DeepSeek-V4 (Flash + Pro) | LLaMA 1/2/3/4, Mistral, Mixtral, DeepSeek (V2/V3), Qwen, Gemma, Phi/PhiMoE, Falcon, Mamba, RWKV, Grok, Command-R, OLMo, Granite, Bitnet, ChatGLM, GLM-Edge, Hunyuan, LFM2, e ~40 altre |
| MoE | ✅ (sqrtsoftplus gate + 8 routed + 1 shared) | ✅ (Mixtral, DBRX, Qwen MoE, PhiMoE, OLMoE, ecc.) |
| MLA (Multi-head Latent Attention) | ✅ nativo V4 (low-rank Q/KV/O, n_groups=8, attn_sink) | ✅ per DeepSeek-V2/V3, non specifico V4 |
| Hyper-Connections | ✅ (hc_mult=4, Sinkhorn collapse) | ❌ |
| MTP / Speculative decoding | ⚠️ Forward block implementato, integrazione decode loop assente | ✅ con draft model separato |
| Multimodale (vision) | ❌ | ✅ LLaVA 1.5/1.6, Qwen2-VL, LFM2-VL, MiniCPM, Moondream, Llama 3.2 Vision |
| Mamba / SSM | ❌ | ✅ Mamba, FalconMamba, RWKV6/7 |
| Embedding models | ❌ | ✅ (BGE, E5, GTE, Nomic, ecc.) |
| Reranking | ❌ | ✅ (BGE-Reranker e simili) |

**Gap**: orizzontalmente enorme. Noi siamo specialisti verticali su V4;
llama.cpp è il "Linux dei runtime LLM".

**Punti dove siamo più avanti**: la nostra implementazione MLA è
specifica per la variante V4 (con `attn_sink`, sliding window,
indexer, compressor con overlap, sparse_attn) — non parità ma
maggiore fedeltà al reference Python di DeepSeek-V4.

---

## 2. Quantizzazione

| | Noi | llama.cpp |
|---|---|---|
| INT8 weight-only | ✅ RTN, gruppo 128, scala F16 | ✅ Q8_0 |
| INT4 weight-only | ✅ RTN, gruppo 128 | ✅ Q4_0, Q4_K_S, Q4_K_M (importance matrix) |
| INT2 | ✅ Sperimentale | ✅ Q2_K + IQ2_XS/XXS |
| 3-bit / 5-bit / 6-bit | ❌ | ✅ Q3_K, Q5_K_S/M, Q6_K |
| FP8 (E4M3) | ✅ Nativo (no conversione richiesta) | ✅ FP8 |
| FP4 (E2M1) | ✅ Nativo per gli expert | ⚠️ MXFP4 (gpt-oss) |
| BF16 / F16 / F32 | ✅ | ✅ |
| INT4 + INT8 calibrato (GPTQ/AWQ/SmoothQuant) | ❌ (in `TODO.md` §0) | ✅ (K-quants usano importance matrix) |
| W8A8 (activations quantizzate) | ❌ (in `TODO.md` §0) | ⚠️ Parziale |

**Gap**: noi RTN-only, llama.cpp ha calibrazione importance-matrix
("K-quants") che recupera 1–2 punti di perplexity rispetto a RTN.

**Punti dove siamo più avanti**: noi leggiamo il checkpoint HF di
DeepSeek-V4 **direttamente** in FP8+FP4 senza conversione, perché V4
nasce così. llama.cpp deve passare per la conversione a GGUF.

---

## 3. Formati file / checkpoint

| | Noi | llama.cpp |
|---|---|---|
| GGUF (de-facto standard) | ❌ | ✅ (formato nativo, con metadata e versioning) |
| SafeTensors (HuggingFace) | ✅ mmap + streaming writer | ⚠️ Solo come input al convertitore |
| GGML legacy | ❌ | ✅ Backward compat |
| Conversione HF → formato runtime | ✅ `converter` CLI (BF16/INT8/INT4/INT2 + sharding allineato per-layer) | ✅ `convert_hf_to_gguf.py` e varianti |
| LoRA adapter loading | ❌ | ✅ Caricamento + merge runtime |

**Gap critico**: senza GGUF siamo tagliati fuori dall'ecosistema
modelli più ampio (Hugging Face TheBloke / unsloth / bartowski hanno
decine di migliaia di GGUF pronti). Una conversione "GGUF → nostro
SafeTensors-like" non è banale perché le quantizzazioni K non
mappano direttamente sui nostri kernel.

---

## 4. Hardware / backend

| | Noi | llama.cpp |
|---|---|---|
| CPU (x86 AVX/AVX2/AVX512/AMX) | ❌ | ✅ |
| CPU (ARM NEON / Apple Accelerate) | ⚠️ Solo CPU reference per test | ✅ |
| CPU (RISC-V RVV/Zvfh) | ❌ | ✅ |
| Metal (Apple Silicon) | ✅ **Nativo, primary target** | ✅ |
| CUDA (NVIDIA) | ❌ | ✅ |
| HIP / ROCm (AMD) | ❌ | ✅ |
| Vulkan | ❌ | ✅ |
| SYCL (Intel / Nvidia) | ❌ | ✅ |
| MUSA (Moore Threads) | ❌ | ✅ |
| CANN (Huawei Ascend) | ❌ | ✅ |
| OpenCL (Adreno) | ❌ | ✅ |
| WebGPU / RPC | ❌ | ⚠️ In progress |
| Inferenza ibrida CPU+GPU | ❌ | ✅ |

**Gap**: scelta architetturale. Siamo deliberatamente Metal-only per
sfruttare la memoria unificata di Apple Silicon (zero-copy mmap →
GPU). Aprire ad altri backend significherebbe riscrivere tutti i 28
kernel `.metal` in altri linguaggi shader o adottare un layer di
astrazione tipo GGML.

---

## 5. Sampling / inferenza

| | Noi | llama.cpp |
|---|---|---|
| Greedy / argmax | ✅ | ✅ |
| Temperature | ✅ | ✅ |
| Top-K | ✅ | ✅ |
| Top-P (nucleus) | ✅ | ✅ |
| Min-P | ❌ | ✅ |
| Mirostat | ❌ | ✅ v1 + v2 |
| Tail-free sampling | ❌ | ✅ |
| Typical sampling | ❌ | ✅ |
| Locally typical | ❌ | ✅ |
| Repetition penalty | ✅ | ✅ |
| Frequency / presence penalty | ❌ | ✅ |
| DRY / XTC samplers | ❌ | ✅ |
| Gumbel-max multinomial | ✅ | ⚠️ Implicito |
| Logit bias / grammar constraints (GBNF) | ❌ | ✅ |
| JSON schema constrained decoding | ❌ | ✅ |

**Gap**: ci mancano min-p (semplice, ~30 LOC), mirostat, e soprattutto
**grammar-constrained decoding (GBNF)** che è una feature killer di
llama.cpp per generare JSON / output strutturati garantiti.

---

## 6. Batching, parallelizzazione, decoding avanzato

| | Noi | llama.cpp |
|---|---|---|
| Batch decode B>1 | ⚠️ Codice shape-compatibile, CLI non lo esercita | ✅ |
| Continuous batching | ❌ | ✅ |
| Parallel decoding multi-prompt | ❌ | ✅ |
| Speculative decoding (draft model) | ⚠️ MTP forward esiste, integrazione no | ✅ |
| KV cache quantization | ❌ | ✅ Q8_0 / Q4_0 sulle K/V |
| KV cache persistence (disk) | ❌ (in `TODO.md` §5) | ✅ `save/load_state` |
| KV cache snapshot in-memory | ✅ `beginDelegation`/`endDelegation` per sub-agent | ⚠️ Solo via prompt cache |
| Sliding window attention | ✅ Win=128 nativo V4 | ✅ |
| Sparse attention | ✅ Con indexer + compressor | ⚠️ Non architettura-specifica |
| FlashAttention | ❌ (in `TODO.md` §5) | ✅ |
| MLA decode multi-token con `startPos>0` | ❌ (in `TODO.md` §5) | N/A |

**Punti dove siamo più avanti**: KV snapshot/restore per
sub-agent delegation è un nostro brevetto di fatto — pensato
specificamente per il caso d'uso "agente principale chiama
sub-agente, torna senza pagare cold prefill". llama.cpp ha una
primitive simile (prompt cache) ma orientata al server multi-utente.

---

## 7. API e interfacce

| | Noi | llama.cpp |
|---|---|---|
| CLI inferenza | ✅ `deepseek` | ✅ `llama-cli` |
| CLI conversione pesi | ✅ `converter` | ✅ `convert_hf_to_gguf.py` |
| HTTP server | ❌ | ✅ `llama-server` (OpenAI-compatible REST + WebUI built-in) |
| OpenAI-compatible API | ⚠️ Solo come **client** verso OpenRouter | ✅ Come server |
| WebUI integrata | ❌ | ✅ |
| GUI desktop nativa | ✅ SwiftUI macOS (chat, agenti, MCP, progetti) | ❌ |
| Bindings Python | ❌ | ✅ `llama-cpp-python` |
| Bindings Node / Go / Rust / Java / Swift / Zig | ❌ | ✅ Tutti |
| CLI perplexity | ❌ | ✅ `llama-perplexity` |
| CLI bench | ❌ | ✅ `llama-bench` |
| CLI embeddings | ❌ | ✅ `llama-embedding` |
| CLI tokenize | ❌ | ✅ `llama-tokenize` |

**Gap critico**: assenza di un **HTTP server OpenAI-compatible**. È
probabilmente il più impattante in termini di "adozione esterna":
chiunque oggi voglia integrare il nostro motore in un'app web o
backend deve scrivere codice Swift, mentre con llama.cpp basta
puntare un endpoint.

**Punti dove siamo più avanti**: GUI desktop nativa con chat, agenti,
MCP, progetti. llama.cpp ha solo una WebUI built-in nel server. Per
l'utente macOS finale, la nostra esperienza è radicalmente migliore.

---

## 8. Tokenizer

| | Noi | llama.cpp |
|---|---|---|
| BPE (HF tokenizer.json) | ✅ | ✅ |
| SentencePiece (.model) | ❌ | ✅ |
| WordPiece | ❌ | ✅ |
| Tiktoken (.tiktoken) | ❌ | ⚠️ Via conversione |
| Auto-detection per modello | ⚠️ Solo BPE | ✅ |
| Byte-level UTF-8 (GPT-2 byte→unicode) | ✅ | ✅ |

**Gap**: BPE-only ci taglia fuori da Mistral, LLaMA, Gemma, ecc. che
usano SentencePiece. Per il caso d'uso V4 è sufficiente.

---

## 9. Chat templates / encoding

| | Noi | llama.cpp |
|---|---|---|
| Template engine | ❌ Hardcoded DSV4 | ✅ Jinja2 + hardcoded per-modello |
| Reasoning blocks (`<think>...</think>`) | ✅ Buffering + UI rendering | ⚠️ Pass-through |
| Tool calls nativi | ✅ DSML `<\|DSML\|tool_calls>` | ✅ Vari formati per modello |
| Tool outputs nativi | ✅ `<\|tool▁outputs▁...\|>` | ✅ |
| Multi-turn conversation | ✅ | ✅ |
| Reverse prompt / stop strings | ⚠️ Solo via EOS | ✅ |
| Function calling OpenAI-style | ✅ (Solo lato client OpenRouter) | ✅ Server-side |

**Gap**: senza Jinja2 dinamico, ogni nuovo modello richiede di
scrivere a mano un nuovo encoding. Per V4-only va bene.

---

## 10. Feature avanzate / ecosistema

| | Noi | llama.cpp |
|---|---|---|
| LoRA / QLoRA adapter loading | ❌ | ✅ Runtime merge |
| Multimodale (vision) | ❌ | ✅ Multiple famiglie |
| Audio input | ❌ | ⚠️ Sperimentale |
| Embeddings | ❌ | ✅ Con pooling modes |
| Classification head | ❌ | ✅ |
| Reranking | ❌ | ✅ |
| GBNF grammar | ❌ | ✅ |
| JSON schema constrained | ❌ | ✅ |
| MCP (Model Context Protocol) | ✅ JSON-RPC stdio + pool + Settings UI | ❌ |
| Multi-agent delegation | ✅ Synthetic `__delegate_to_agent` + KV snapshot, nesting 3 livelli, cycle prevention | ❌ |
| Agent presets (system + tools + sampling) | ✅ `AgentLibrary` + UI | ❌ |
| Project-attached chats (RAG con delimiter nativi) | ✅ `ProjectIndexer` | ❌ |
| Remote inference fallback (OpenRouter) | ✅ Streaming SSE + cost tracking | ❌ |
| Crash recovery (`pendingTurn`) | ✅ Local chats | ❌ |

**Punti dove siamo molto più avanti**: tutta la colonna del "client
agentico" è nostra. MCP + multi-agent + projects + KV snapshot per
delegation + OpenRouter fallback **non esistono** in llama.cpp.
llama.cpp è puro runtime, noi siamo runtime + client.

---

## 11. Build / dipendenze / piattaforme

| | Noi | llama.cpp |
|---|---|---|
| Build system | Swift Package Manager + xcodegen | CMake + Make |
| Dipendenze esterne | **Zero** (solo Foundation + Metal) | Single-header (httplib, json, stb, miniaudio) |
| Cross-platform | macOS only | Linux / macOS / Windows / iOS / Android / Docker |
| Container support | ❌ | ✅ Docker ufficiale |
| CI/CD pre-built binaries | ❌ | ✅ GitHub Releases multi-platform |
| Distribuzione GUI | Xcode `.app` | N/A (server) |

---

## 12. Performance e ottimizzazioni

| | Noi (stato attuale) | llama.cpp |
|---|---|---|
| GEMM con simdgroup_matrix | ❌ Scalar tile (~1-2 TFLOPS vs ~10 teorici) | ✅ Per-backend |
| FlashAttention tiling | ❌ One-thread-per-cell | ✅ |
| Persistent MoE kernel | ❌ N dispatch tiny + CPU sync | ✅ Batched |
| Pipeline state caching | ❌ Lazy per init | ✅ |
| NUMA awareness | N/A (Apple Silicon non NUMA) | ✅ |
| Page-cache prefetch | ❌ (ROADMAP) | ⚠️ |

La nostra `PERFORMANCE.md` stima 5–10× di speedup latente solo da
`simdgroup_matrix` sulle Linear, più 3–5× da FlashAttention.
Attualmente il progetto è **correctness-first**, non
performance-first. llama.cpp ha 4 anni di tuning kernel.

---

## 13. Testing e quality metrics

| | Noi | llama.cpp |
|---|---|---|
| Unit test kernel | ✅ 22 XCTest (Metal vs CPU reference) | ✅ Test suite C++ |
| Test tokenizer | ✅ `BPETokenizerTests` | ✅ |
| Test sampler | ❌ (in `TODO.md` §7) | ✅ |
| Test encoder/chat template | ❌ (in `TODO.md` §7) | ✅ |
| Test end-to-end vs reference Python | ❌ Bloccato (richiede CUDA) | ⚠️ Vs HF transformers |
| Perplexity benchmark | ❌ | ✅ `llama-perplexity` |
| Throughput benchmark | ❌ | ✅ `llama-bench` |

---

## 14. Riepilogo strategico

### Cosa **non possiamo permetterci di ignorare** (gap critici per la nostra missione)

1. **HTTP server OpenAI-compatible**. Senza questo, l'unico modo per
   un'app esterna di usarci è linkare `DeepSeekKit` come libreria
   Swift. Bloccante per qualsiasi adozione fuori dall'app SwiftUI.
   Sforzo stimato: ~2 settimane (server SSE + endpoint
   `/v1/chat/completions`).

2. **Grammar-constrained decoding (GBNF) o almeno JSON schema**.
   Riusiamo `tool_calls` per ottenere JSON strutturato ma è
   probabilistico. GBNF lo rende garantito. Sforzo: ~1 settimana per
   un subset.

3. **Min-P sampling**. Trivial (~30 LOC) e ormai standard.

4. **Calibrazione GPTQ/AWQ per INT4/INT8**. Già in `TODO.md` §0.
   Senza, perdiamo 1–2 punti di perplexity sulle quantizzazioni
   aggressive.

5. **MLA multi-token con `startPos > 0`** + **simdgroup_matrix GEMM**.
   Già in `TODO.md` §5, sono i due single biggest perf win.

### Cosa **possiamo deliberatamente ignorare** (out-of-scope per la nostra missione)

- Supporto multi-architettura (LLaMA, Mistral, Gemma…). Siamo
  V4-specialisti per design.
- Backend non-Metal. Siamo Apple-specialisti per design.
- Multimodale vision. Né V4-Flash né V4-Pro hanno vision encoder
  nello scope attuale.
- LoRA. Caso d'uso "fine-tuning + serving" non è nel nostro target.
- Embeddings / reranking. Sono modelli diversi, non rilevanti.
- Bindings Python / Node / Go. Se serve, l'HTTP server li
  rimpiazza tutti.

### Cosa abbiamo che llama.cpp **non avrà mai** (per design)

- GUI desktop nativa macOS con MCP, agenti, projects, delegation.
- Integrazione OpenRouter come fallback remoto trasparente.
- KV snapshot/restore per sub-agent (caso d'uso agentico).
- Encoding DSV4 1:1 con il reference Python di DeepSeek.
- FP8/FP4 nativo dal checkpoint HF senza conversione.

---

## 15. Posizionamento

| Asse | Noi | llama.cpp |
|---|---|---|
| **Profondità** (un modello fatto bene) | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| **Ampiezza** (tanti modelli, tante piattaforme) | ⭐ | ⭐⭐⭐⭐⭐ |
| **Esperienza utente end-user** (GUI) | ⭐⭐⭐⭐⭐ | ⭐ |
| **Esperienza developer** (API, bindings, server) | ⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Performance per token** | ⭐⭐ (correctness-first) | ⭐⭐⭐⭐⭐ |
| **Ecosistema modelli pre-quantizzati** | ⭐ (solo V4) | ⭐⭐⭐⭐⭐ (GGUF hub) |
| **Feature agentiche / tool calling avanzato** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |

**Posizionamento sintetico**: siamo "the Mac app for DeepSeek-V4 with
serious agentic chops". llama.cpp è "the universal local LLM
runtime". Sono **complementari, non sostituti**. Un utente sofisticato
potrebbe ragionevolmente avere entrambi installati.

---

## 16. Azioni proposte (in ordine di ROI decrescente)

1. **HTTP server con `/v1/chat/completions` OpenAI-compatible** —
   sblocca integrazione esterna.
2. **`simdgroup_matrix` BF16 GEMM** — 5–10× su ogni Linear.
3. **MLA multi-token con `startPos > 0`** — 5–10× sui turn tool-heavy.
4. **Min-P + frequency/presence penalty** — quick wins di parità
   sampling.
5. **Grammar-constrained decoding (subset JSON schema)** — feature
   killer per output strutturati.
6. **GPTQ/AWQ calibrato per INT4/INT8** — quality win sulle
   quantizzazioni aggressive.
7. **`llama-bench`-equivalente CLI** — finalmente possiamo dire
   "X tokens/sec" con un numero riproducibile.
8. **KV cache persistence to disk** — già in roadmap (B3).
9. **FlashAttention tiling per sparse_attn** — 3–5× sull'attention.

Punti 2, 3, 8, 9 sono già in `TODO.md`/`ROADMAP.md`. Punti 1, 4, 5, 6,
7 sono **nuovi** che emergono dal confronto con llama.cpp.
