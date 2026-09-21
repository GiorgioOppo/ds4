# Four-model integration contracts

Research date: 2026-09-21. C source pinned to `acf5c16bb6a01f8c4f027a0db265ae270d8907ab`; initial Swift inspection at `b4a9ee67`. These notes preserve the source contracts and artifact manifests used for the port. Current implementation and validation limits are documented in [NATIVE-SWIFT-MODELS.md](NATIVE-SWIFT-MODELS.md). No published model weights were downloaded; numerical tests use synthetic fixtures. The implementation is entirely Swift/Metal; C is a reference, not a runtime dependency.

## Immutable artifact proposals

The following exact filenames come from upstream `download_model.sh:6–41` and `docs/BONSAI.md:14–65`. Repository revisions, byte sizes and LFS SHA-256 values were read from public Hugging Face JSON manifests. These describe remote objects, not downloaded/locally verified bytes. Resolve each file using `https://huggingface.co/<repository>/resolve/<revision>/<filename>`; never silently replace these revisions with `main`.

### prism-ml/Ternary-Bonsai-2-27B-gguf

Revision: `6ed5e12bf84b7a63069882c91dd9e9218647d17b`. [Manifest](https://huggingface.co/api/models/prism-ml/Ternary-Bonsai-2-27B-gguf/revision/6ed5e12bf84b7a63069882c91dd9e9218647d17b?blobs=true).

| Artifact | Bytes | SHA-256 |
|---|---:|---|
| `Ternary-Bonsai-2-27B-F16.gguf` | 53808408928 | `f6f3b2c9b41956c34b379ec7c301dc936bc38d79b3c24c83388dd7d76000c180` |
| `Ternary-Bonsai-2-27B-PQ2_0.gguf` | 7206168928 | `3907dc1658db1f78a9826bf8d5bcb8dc65db0d466388937af57f2294fae62ec1` |
| `Ternary-Bonsai-2-27B-PTQ1_0.gguf` | 5946648928 | `53107f530aa52eb00912263ab1ee29bd199261c87cd7b4ad4ca1318c1fe33ee3` |
| `Ternary-Bonsai-2-27B-mmproj-BF16.gguf` | 931145856 | `e287342d92332fa3577ed1d42e921dac9370c08da58ba9337fa450f6cc76cfd7` |
| `Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf` | 629246976 | `6807ede61d570bb86ba34b756a0fa109edc33668604de867c6ea6d8f1d631903` |

### antirez/qwen3.8-flash-next-gguf

Revision: `d600fe1a43d2e1cdcadb85144ce3142f66f9eefe`. [Manifest](https://huggingface.co/api/models/antirez/qwen3.8-flash-next-gguf/revision/d600fe1a43d2e1cdcadb85144ce3142f66f9eefe?blobs=true).

| Artifact | Bytes | SHA-256 |
|---|---:|---|
| `Qwen3.8-Flash-Next-Q2.gguf` | 147207127040 | `b1b93fa69aca5f187b0fb813aca8f3ec1beb5cf8cf0bd38cf041b93e0b6ccac9` |
| `Qwen3.8-Flash-Next-Q4.gguf` | 177280286720 | `680944460a8cbe93ba8b6d7b6107213ffb7e22320bd913000e563ca0a0f25a8a` |

### ggml-org/Qwen3.8-Flash-Next-GGUF

Revision: `01534bc2e1877d5de995b73d247d4459d273e688`. [Manifest](https://huggingface.co/api/models/ggml-org/Qwen3.8-Flash-Next-GGUF/revision/01534bc2e1877d5de995b73d247d4459d273e688?blobs=true).

| Artifact | Bytes | SHA-256 |
|---|---:|---|
| `mmproj-Qwen3.8-Flash-Next-Q8_0.gguf` | 616703104 | `b2e9b5e4a44c107f8867e67dbf09b607fd99ae33c1a97a60a6720aeb252a9dad` |

### antirez/deepseek-v4.1-flash-gguf

Revision: `dd8a266f7145edc19e2334b46e19b6821f221dc7`. [Manifest](https://huggingface.co/api/models/antirez/deepseek-v4.1-flash-gguf/revision/dd8a266f7145edc19e2334b46e19b6821f221dc7?blobs=true).

| Artifact | Bytes | SHA-256 |
|---|---:|---|
| `DeepSeek-V4.1-Flash-Q2.gguf` | 365713686528 | `1ce6a8f8806205c13330d7ca287bd198331dc5ca35ccc5d8a9a92a188a6f6f42` |
| `DeepSeek-V4.1-Flash-Q4.gguf.part1` | 480000000000 | `6442b1f9224079662c02003c0ef9ef6be6e2aff509510f681dab9e6cc41df246` |
| `DeepSeek-V4.1-Flash-Q4.gguf.part2` | 38596067328 | `7c3e10646c918eeaffbc39305a75ec96117450262c61454ff194cef00d7617f0` |
| `DeepSeek-V4.1-Flash-Vision.gguf` | 970555552 | `cc283f032b3e8b8d78aeb5fccaa14e97b859b0c53aae3cd6bffa690ddf0e9e15` |

### antirez/glm-5.3-flash-gguf

Revision: `b2fa29d7a6b410db11221c904973967b80b760f5`. [Manifest](https://huggingface.co/api/models/antirez/glm-5.3-flash-gguf/revision/b2fa29d7a6b410db11221c904973967b80b760f5?blobs=true).

| Artifact | Bytes | SHA-256 |
|---|---:|---|
| `GLM-5.3-Flash-FP8.gguf` | 327209059584 | `59275e79a5246835226230616b3865fb599c661f242c38316b1cf82869bd14c9` |
| `GLM-5.3-Flash-Q2.gguf` | 96505816384 | `e81fd6241c6e55a64e1e14e47a3eab61a173fa8d7e4b5c1d1848827119705b32` |
| `GLM-5.3-Flash-Q4_K.gguf` | 190875526464 | `c7a0d950363238dd7804782c88340d737775aba53a15f8d4fdcc34e984f25221` |
| `GLM-5.3-Flash-Vision-Encoder.gguf` | 1127280960 | `ae23e14c6979e889051b2e4a39351abcdafb161e18e606fae4d8c40095a4bf3a` |

DeepSeek4.1 Q4 is distributed as two **byte fragments**, not two independent GGUF shards. Concatenated size `518596067328`, SHA-256 `a5e2e2c3ada4b2e98d9f9e4b50f6d9c2a12c2c96f5da165c07e13aff9264984e`, from pinned `download_model.sh:464–474,594–671`. Do not feed `.part1` to an ordinary GGUF loader or mark fragments selectable. Q2 is the simplest first supported catalog recipe. Excluded GLM Flash FP8 is packaged but not implemented even in C; excluded Bonsai F16 is not admitted by its folded PQ2/PTQ binding. Qwen's ggml-org repository is an encoder source here, not an alternative language checkpoint recommendation.

## Runtime identities and memory contracts

| Requested model | GGUF architecture | Required runtime distinctions |
|---|---|---|
| Bonsai 2 27B | `qwen35` | Accept only validated Prism metadata/shape: 64 layers, width5120, FFN17408, GDN×3 then GQA, folded Hadamard1024, PQ2/PTQ. No MoE/HC/MTP. Text CPU reference or Metal upstream; images Metal with matching projector. |
| Qwen3.8 Flash Next | `qwen4exp` | 48 trunk layers plus optional embedded predictor; width2560, HC4, GDN/GQA+sparse indexer, MoE512/top10, original BF16 n-grams on SSD. Q2 down logical640 padded768; Q4 down MXFP4. Metal or one CUDA GPU upstream; Swift port uses Metal. |
| DeepSeek4.1 Flash | `deepseek41` | Separate graph, tokenizer and Engram SSD lookup; metadata uses `num_hidden_layers`, `hidden_size`, `vocab_size`, `max_position_embeddings`, not the ordinary DeepSeek4 namespace schema. Encoder differs from V4 Vision-Exp. |
| GLM5.3 Flash | `glm5-next` | KDA recurrence plus DSA, HC and embedded MTP. Distinct from full GLM5.3/GLM5.2 `glm-dsa`. GLM-family tokenizer can share existing primitives after parity checks. |

Bonsai PQ2 is 6.71GiB, PTQ about5.54GiB, plus context/GDN/scratch/encoder. Qwen Q2 is137.10GiB on disk but41.73GiB main/MTP; Q4 is165.11GiB/69.74GiB; BF16 n-gram table stays on SSD. DeepSeek4.1 Q2 is341GiB disk/152GiB main plus189GiB SSD Engram; Q4 is483GiB/294GiB main. GLM5.3 Flash Q2 is90GiB, Q4~178GiB. Disk size is not resident memory and no catalog entry should promise fit solely from GGUF size.

Sources: [model dispatch](https://github.com/GiorgioOppo/ds4/blob/acf5c16bb6a01f8c4f027a0db265ae270d8907ab/ds4.c#L7274), [runtime option gates](https://github.com/GiorgioOppo/ds4/blob/acf5c16bb6a01f8c4f027a0db265ae270d8907ab/ds4.c#L71235), [models](https://github.com/GiorgioOppo/ds4/blob/acf5c16bb6a01f8c4f027a0db265ae270d8907ab/docs/MODELS.md), [Bonsai binding](https://github.com/GiorgioOppo/ds4/blob/acf5c16bb6a01f8c4f027a0db265ae270d8907ab/bonsai_bind.inc).

## Shared Swift integration points

- `Sources/DS4Core/Formats/GGUF/GGUFTypes.swift:55`: add **inspection** descriptors `39:mxfp4,32,17`, `142:pq2_0,128,34`, `143:ptq1_0,128,28`, following `ds4.c:2466–2468`. This permits truthful tensor sizes; it does not enable compute. Current unknown types yield zero bytes in `GGUFModel.swift:164` and skip extent validation at:181. Preserve checked arithmetic and distinguish unknown type from an actually empty tensor.
- `Sources/DS4Core/Model/Common/ModelArchitecture.swift`: explicit IDs/namespaces for new architectures. `glm5-next` cannot use normalized `glm5next.*` metadata. `qwen35` alone must not label every Qwen3.5 model Bonsai: validate Prism signature, shape, tokenization and transform tables. Keep recognition separate from implementation availability.
- `Sources/DS4Engine/Runtime/Common/ModelInspector.swift:11–23`: exact metadata fallback map, especially DeepSeek4.1 nonstandard field names; vocabulary can require tokenizer token-array length when no explicit vocab key. Gate-aware availability and capabilities must route all four explicitly instead of inheriting DeepSeek controls by family prefix.
- `Sources/DS4Core/Tokenization/API/TokenizerFactory.swift`: add native Qwen35 BPE/ChatML for Qwen+Bonsai and V4.1 tokenization/prompt contract. C `ds4.c:43309–43316,43441–43465` shares Qwen pretokenizer but Bonsai excludes Flash Next reasoning-effort system instructions. Do not call DeepSeek tokenizer as a generic fallback. GLM Flash should share GLM primitives only after tokenizer/template fixture equality.
- `Sources/DS4Engine/Runtime/Common/BackendSelector.swift`: add concrete backend cases after native graph implementation; keep existing DeepSeek/GLM/Laguna behavior. No `runnable` catalog flag that still reaches a placeholder decoder.
- `Sources/DS4Engine/ModelManagement/Catalog/ModelCatalog.swift`: new stable IDs/profiles and `HuggingFaceSource` constants using the manifest pins above. Existing `ModelTarget` already accepts source/revision/SHA/size. Add separate catalog files for each family; encoder artifacts have `.optionalComponent`, not `.mainModel`.
- Generalize encoder associations currently tied to `requiresVisionEncoder` and `DeepSeekV4VisionCatalog.encoder`. Text checkpoints for Qwen/GLM/DeepSeek4.1/Bonsai run without an encoder; image capability requires the matching family/profile encoder. Never replace the selected language model with a projector.
- `Sources/DS4Engine/ModelManagement/Download/ModelDownloader.swift:46–54,342–363` already supports pinned targets; reuse checksum/size checks. Native bounded/paged SSD access is required for Qwen n-grams and DeepSeek4.1 Engram. Metadata inspection should pass `metalMapping:false,prefetchCPU:false`.
- GUI scanning uses `ModelCatalogRegistry.selectableEntries` in `Sources/DwarfStar/Features/ModelManagement/Models/ModelCatalog.swift`; this derives selectable files from actual runtime gates. Memory estimates and controls must distinguish dense/recurrent/Bonsai from MoE/DeepSeek tuning.

## Model-free regression fixtures

1. GGUF type table:142→128/34,143→128/28,39→32/17; multiple blocks, overflow, truncated payload and unknown-type behavior. Do not test only rounded totals: binders must reject non-block-aligned row dimensions where their kernels require alignment.
2. Minimal synthetic GGUF metadata for each exact architecture; preserved wire namespace; missing/cross-family keys; explicit `qwen35` without Prism metadata not falsely accepted as Bonsai; V4.1 field names recognized without reading token payloads.
3. Catalog snapshot: exact IDs, revisions, filenames, byte lengths, SHA256; URL resolution; encoder exclusion from primary/selectable model entries; model/encoder mismatch rejection; Q4 byte fragments never interpreted as GGUF shards.
4. Tokenizer golden fixtures from upstream tests: Unicode combining marks, CJK, spaces/newlines, numbers, ChatML special markers, thinking enabled/disabled, tool messages and vision placeholder IDs. Separate Bonsai prompt fixtures from Flash Next reasoning effort.
5. Native decoder graph fixtures: full versus incrementally evaluated suffix, reset/replay, context overflow before mutation, cancellation leaving a clearly resettable state, batch tails and recurrent state parity. Qwen additionally n-gram EOS/hash/history, SSD offset bounds and padded MoE down dimension; GLM Flash KDA+DSA alternation; V4.1 Engram/source-layer sharing; Bonsai inverse/folded Hadamard and PQ2/PTQ exact codec.

Existing fixture patterns: `Tests/DS4CoreTests/Core/Formats/GGUF/GGUFTests.swift`, `Core/Tokenization/API/TokenizerFactoryTests.swift`, `Core/Model/Common/ModelArchitectureTests.swift`, `Engine/Runtime/ModelInspectorGLM52Tests.swift`, `Engine/ModelManagement/DeepSeekV4VisionCatalogTests.swift`. Model-free success establishes contracts and primitive numerics; it does not establish real-model quality or end-to-end throughput.

## Download implementation

The catalog includes the four text families, matching image encoder downloads, the reference Bonsai F16 and GLM Flash FP8. F16, FP8 and all five new image encoder variants are explicitly download-only; downloading them cannot change the active text model. Their manifests were re-read on 2026-09-21 at the immutable revisions above. The Qwen encoder is published by ggml-org; the other new artifacts use antirez or Prism.

DeepSeek V4.1 Q4 downloads the two ordered byte fragments and assembles a single local `DeepSeek-V4.1-Flash-Q4.gguf`. The assembler uses at most 8 MiB per I/O buffer, checks each fragment SHA-256 plus the concatenated SHA-256, synchronizes staging, and publishes by same-directory rename. A cancelled operation keeps `.assembling.part`; retry compares every existing prefix byte with the source fragments before appending. Bad prefixes, checksums, regular-file checks or space checks leave the final model unpublished. Source fragments are kept, so peak disk use is 1,037,192,134,656 bytes plus the space reserve. The UI shows this requirement before download. Only the assembled GGUF appears as a model choice; individual `.part1`/`.part2` paths are never selectable from the catalog.

`ModelAssemblyTests` uses tiny synthetic local fragments to exercise merge, cancellation, safe resume, source/final checksums, staging symlink refusal and catalog boundaries. It downloads no model payloads. Full-model inference and throughput remain separate validation steps.

### Additional antirez download (full GLM 5.3)

Fresh public manifest and revision-specific manifest verified on 2026-09-21: [antirez/glm-5.3-gguf](https://huggingface.co/api/models/antirez/glm-5.3-gguf/revision/1434dad1a25c6318c9befef38d7735a1e0e15f1d?blobs=true). Revision `1434dad1a25c6318c9befef38d7735a1e0e15f1d`, artifact `GLM-5.3-UD-IQ2_XXS_RoutedIQ2XXS_blk78Q2K.gguf`, 211075860864 bytes, SHA-256 `059b36accd4c9acf73099da9f703b574d627869d619b7c4c316aa856e33d472e`. This is the full model, not GLM 5.3 Flash, and is explicitly download-only in the catalog. No model payload was downloaded.
